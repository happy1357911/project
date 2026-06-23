import argparse
import copy
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader, Dataset

from mtl_meta_irl_transformer import MTLMetaIRLTransformer


SUPPORTED_CC = {(5,0),(6,0),(6,1),(7,0),(7,5),(8,0),(8,6),(9,0),(8,9)}


def safe_pick_device():
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) not in SUPPORTED_CC:
                print(f"[MetaIRL v7.4] CUDA sm_{major}{minor} not fully supported, fallback to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
    except Exception as e:
        print(f"[MetaIRL v7.4] safe_pick_device error: {e}")
    return torch.device("cpu")


DEVICE = safe_pick_device()

LR = 5e-4  #1e-4 -> 1e-5
BATCH = 32 
EPOCHS = 40  #40 -> 80
PATIENCE = 8
WEIGHT_DECAY = 1e-4
DEFAULT_REWARD_WEIGHT = 0.15
GRAD_CLIP = 1.0

TASK_INFO = {
    "Task1": {
        "csv": "task1_all_three_common14_v1.csv",
        "label_cols": [
            "hearing_degree_WHO_PTAbased_right",
            "hearing_degree_WHO_PTAbased_left",
        ],
        "feature_cols": [
            "right_500Hz","right_1000Hz","right_2000Hz","right_4000Hz",
            "left_500Hz","left_1000Hz","left_2000Hz","left_4000Hz",
            "right_PTA","left_PTA",
        ],
    },
    "Task2": {
        "csv": "task2_real_clean_with_BC_type_v1.csv",
        "label_cols": [
            "hearing_type_right",
            "hearing_type_left",
        ],
        "feature_cols": [
            "right_500Hz","right_1000Hz","right_2000Hz","right_4000Hz",
            "left_500Hz","left_1000Hz","left_2000Hz","left_4000Hz",
            "bc_right_500Hz","bc_right_1000Hz","bc_right_2000Hz","bc_right_4000Hz",
            "bc_left_500Hz","bc_left_1000Hz","bc_left_2000Hz","bc_left_4000Hz",
            "right_PTA","left_PTA",
        ],
    },
    "Task3": {
        "csv": "task3_real_clean_PTA_degree_v1.csv",
        "label_cols": [
            "tymp_right_type",
            "tymp_left_type",
        ],
        "feature_cols": [
            "right_500Hz","right_1000Hz","right_2000Hz","right_4000Hz",
            "left_500Hz","left_1000Hz","left_2000Hz","left_4000Hz",
            "right_PTA","left_PTA",
            "tymp_right_Vea","tymp_right_peak_daPa","tymp_right_peak_mmho","tymp_right_Width_daPa",
            "tymp_left_Vea","tymp_left_peak_daPa","tymp_left_peak_mmho","tymp_left_Width_daPa",
        ],
    },
}

#整理資料(打上特徵標籤)
class MultiLabelTabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[self.feature_cols].astype(np.float32).values
        ys = [int(row[lbl]) for lbl in self.label_cols]
        raw = row.to_dict()
        return torch.tensor(x, dtype=torch.float32), [torch.tensor(y, dtype=torch.long) for y in ys], raw


def collate_with_raw(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    n_labels = len(batch[0][1])
    ys = [torch.stack([item[1][j] for item in batch], dim=0) for j in range(n_labels)]
    raws = [item[2] for item in batch]
    return xs, ys, raws

#固定seed(確保重現性)
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#去空格
def sanitize_column_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s+", "_", x)
    return x

#轉資料形式(並且確保沒空值)
def clean_value(v):
    try:
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def pad_and_clean(df: pd.DataFrame, union_features: List[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [sanitize_column_name(c) for c in df.columns]
    for c in union_features:
        if c not in df.columns:
            df[c] = 0.0
    for c in union_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def clean_label_columns(df: pd.DataFrame, label_cols: List[str]):
    out = df.copy()
    missing_stats = {}
    for lbl in label_cols:
        if lbl not in out.columns:
            raise KeyError(f"Missing label column: {lbl}")
        s = out[lbl].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan, "NULL": np.nan})
        out[lbl] = s
        missing_stats[lbl] = int(out[lbl].isna().sum())
    before = len(out)
    out = out.dropna(subset=label_cols).reset_index(drop=True)
    missing_stats["rows_before"] = int(before)
    missing_stats["rows_after"] = int(len(out))
    missing_stats["rows_dropped_any_label_missing"] = int(before - len(out))
    return out, missing_stats


def split_dataframe(df: pd.DataFrame, task_name: str, seed: int):
    label_cols = TASK_INFO[task_name]["label_cols"]
    strat_col = label_cols[0]
    work = df.copy()
    rare_summary = {}
    force_train_idx = set()

    vc = work[strat_col].value_counts()
    rare_classes = [k for k, v in vc.items() if v < 2]
    for cls in rare_classes:
        idxs = work.index[work[strat_col] == cls].tolist()
        force_train_idx.update(idxs)
        rare_summary[str(cls)] = len(idxs)

    forced_df = work.loc[sorted(list(force_train_idx))].copy()
    rest_df = work.drop(index=list(force_train_idx)).copy()

    if len(rest_df) >= 10 and (not rest_df[strat_col].isna().any()):
        vc2 = rest_df[strat_col].value_counts()
        if len(vc2) > 1 and vc2.min() >= 2:
            train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed, stratify=rest_df[strat_col])
        else:
            train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed)
    elif len(rest_df) >= 2:
        train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed)
    else:
        train_df = rest_df.copy()
        val_df = rest_df.iloc[:0].copy()

    train_df = pd.concat([train_df, forced_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df, rare_summary


def compute_norm_meta(train_df: pd.DataFrame, feature_cols: List[str]):
    mu = {}
    sigma = {}
    for c in feature_cols:
        m = float(train_df[c].mean())
        s = float(train_df[c].std(ddof=0))
        if not np.isfinite(s) or s < 1e-8:
            s = 1.0
        mu[c] = m
        sigma[c] = s
    return {"mu": mu, "sigma": sigma}


def apply_norm(df: pd.DataFrame, feature_cols: List[str], mu: Dict[str, float], sigma: Dict[str, float]):
    out = df.copy()
    for c in feature_cols:
        out[c] = (out[c].astype(float) - mu[c]) / sigma[c]
    return out


def build_class_weight_tensor(y_series: pd.Series, n_classes: int):
    counts = y_series.value_counts().to_dict()
    total = len(y_series)
    weights = []
    for i in range(n_classes):
        c = counts.get(i, 0)
        if c <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (n_classes * c))
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    return w


def expert_consistency_score(task_name: str, label_col: str, row: dict, pred_name: str):
    side = "right" if "right" in label_col.lower() else "left"
    p = str(pred_name).upper()
    if task_name == "Task1":
        pta = clean_value(row.get(f"{side}_PTA", 0.0))
        if pta < 20 and "NORMAL" in p: return 1.0
        if 20 <= pta < 35 and "MILD" in p: return 1.0
        if 35 <= pta < 50 and "MODERATE" in p: return 1.0
        if 50 <= pta < 65 and ("MODERATELY" in p or "SEVERE" in p): return 1.0
        if 65 <= pta < 80 and "SEVERE" in p: return 1.0
        if pta >= 80 and ("PROFOUND" in p or "COMPLETE" in p): return 1.0
        return -0.2
    elif task_name == "Task2":
        ac = np.mean([clean_value(row.get(f"{side}_500Hz", 0.0)),
                      clean_value(row.get(f"{side}_1000Hz", 0.0)),
                      clean_value(row.get(f"{side}_2000Hz", 0.0)),
                      clean_value(row.get(f"{side}_4000Hz", 0.0))])
        bc = np.mean([clean_value(row.get(f"bc_{side}_500Hz", 0.0)),
                      clean_value(row.get(f"bc_{side}_1000Hz", 0.0)),
                      clean_value(row.get(f"bc_{side}_2000Hz", 0.0)),
                      clean_value(row.get(f"bc_{side}_4000Hz", 0.0))])
        abg = ac - bc
        if abg >= 15 and ("CHL" in p or "MHL" in p or "MIHL" in p): return 1.0
        if abg < 10 and ("SNHL" in p or "WNL" in p or "TONE LOSS" in p): return 0.8
        return -0.2
    elif task_name == "Task3":
        peak = clean_value(row.get(f"tymp_{side}_peak_daPa", 0.0))
        cmpl = clean_value(row.get(f"tymp_{side}_peak_mmho", 0.0))
        width = clean_value(row.get(f"tymp_{side}_Width_daPa", 0.0))
        if abs(peak) <= 100 and 0.2 <= cmpl <= 1.8 and width <= 200 and p == "A": return 1.0
        if cmpl < 0.15 and p == "B": return 1.0
        if peak < -100 and p == "C": return 1.0
        return -0.2
    return 0.0


def build_support(X: torch.Tensor, ys: List[torch.Tensor], label_cols: List[str], max_per_class: int = 2):
    support = {}
    for lbl, y in zip(label_cols, ys):
        chosen = []
        y_cpu = y.detach().cpu().numpy()
        for c in sorted(set(y_cpu.tolist())):
            idxs = np.where(y_cpu == c)[0][:max_per_class]
            chosen.extend(idxs.tolist())
        if len(chosen) == 0:
            support[lbl] = (None, None)
        else:
            uniq = sorted(set(chosen))
            support[lbl] = (X[uniq], y[uniq])
    return support


def safe_multiclass_auc(y_true, prob, n_classes: int):
    try:
        if len(set(y_true)) < 2:
            return None
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.shape[1] == 1:
            return None
        return float(roc_auc_score(y_bin, prob, average="macro", multi_class="ovr"))
    except Exception:
        return None


def evaluate_task(model, loader, meta, task_name, criteria_dict):
    model.eval()
    label_cols = meta["tasks"][task_name]["label_cols"]
    class_names = meta["tasks"][task_name]["class_names"]

    all_true = {lbl: [] for lbl in label_cols}
    all_pred = {lbl: [] for lbl in label_cols}
    all_prob = {lbl: [] for lbl in label_cols}

    total_loss = 0.0
    with torch.no_grad():
        for X, ys, raws in loader:
            X = X.to(DEVICE)
            ys = [y.to(DEVICE) for y in ys]
            logits_dict, reward_dict = model(X, task_name, support=None)
            batch_loss = 0.0
            for lbl, y in zip(label_cols, ys):
                logits = logits_dict[lbl]
                loss = criteria_dict[lbl](logits, y)
                batch_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_true[lbl].extend(y.cpu().tolist())
                all_pred[lbl].extend(preds.cpu().tolist())
                all_prob[lbl].extend(probs.cpu().tolist())
            total_loss += batch_loss

    metrics = {"loss": total_loss / max(len(loader), 1)}
    cm_store = {}
    for lbl in label_cols:
        y_true = all_true[lbl]
        y_pred = all_pred[lbl]
        probs = np.array(all_prob[lbl], dtype=float) if len(all_prob[lbl]) else np.zeros((0, len(class_names[lbl])))
        n_classes = len(class_names[lbl])
        metrics[f"{lbl}_acc"] = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        metrics[f"{lbl}_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
        metrics[f"{lbl}_balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0
        auc = safe_multiclass_auc(y_true, probs, n_classes) if len(y_true) else None
        metrics[f"{lbl}_macro_auc_ovr"] = auc
        labels = list(range(n_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels) if y_true else np.zeros((len(labels), len(labels)), dtype=int)
        cm_store[lbl] = cm.tolist()
    return metrics, cm_store


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_experiment(data_dir: str, out_root: str, seed: int, exp_name: str):
    seed_everything(seed)
    reward_weight = 0.0 if exp_name == "no_irl" else DEFAULT_REWARD_WEIGHT
    enable_meta = False if exp_name == "no_meta" else True
    task_sequence = ["Task1", "Task2", "Task3"] if exp_name != "single_task" else ["Task1", "Task2", "Task3"]

    run_dir = Path(out_root) / f"{exp_name}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MetaIRL v7.4] Using device: {DEVICE}")
    print(f"[MetaIRL v7.4] Run dir: {run_dir}")
    print(f"[MetaIRL v7.4] Loading CSVs from: {Path(data_dir).resolve()}")

    raw_dfs = {}
    union_features = set()
    for task_name, info in TASK_INFO.items():
        csv_path = Path(data_dir) / info["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {task_name}: {csv_path.resolve()}")
        df = pd.read_csv(csv_path)
        raw_dfs[task_name] = df
        union_features.update(info["feature_cols"])
    union_features = sorted(list(union_features))
    print(f"[MetaIRL v7.4] Union features = {len(union_features)}")

    tasks_meta = {}
    task_label_class_counts = {}
    loaders_train = {}
    loaders_val = {}
    criteria_by_task = {}
    norm_meta = {}
    train_sizes = {}
    cleaning_log = {}

    for task_name, info in TASK_INFO.items():
        df = pad_and_clean(raw_dfs[task_name], union_features)
        df, missing_stats = clean_label_columns(df, info["label_cols"])
        cleaning_log[task_name] = {"missing_label_stats": missing_stats}
        if len(df) < 2:
            raise ValueError(f"{task_name} has too few rows after dropping missing labels: {len(df)}")

        train_df, val_df, rare_summary = split_dataframe(df, task_name, seed)
        cleaning_log[task_name]["rare_classes_forced_to_train"] = rare_summary

        label_class_names = {}
        label_num_classes = {}
        criteria_dict = {}
        for lbl in info["label_cols"]:
            le = LabelEncoder()
            le.fit(df[lbl].astype(str))
            train_df.loc[:, lbl] = le.transform(train_df[lbl].astype(str))
            val_df.loc[:, lbl] = le.transform(val_df[lbl].astype(str))
            label_class_names[lbl] = list(le.classes_)
            label_num_classes[lbl] = len(le.classes_)
            class_weights = build_class_weight_tensor(train_df[lbl], label_num_classes[lbl])
            criteria_dict[lbl] = nn.CrossEntropyLoss(weight=class_weights)

        task_label_class_counts[task_name] = label_num_classes
        tasks_meta[task_name] = {
            "label_cols": info["label_cols"],
            "class_names": label_class_names,
            "num_classes": label_num_classes,
        }
        criteria_by_task[task_name] = criteria_dict

        nm = compute_norm_meta(train_df, union_features)
        norm_meta[task_name] = nm
        train_df_n = apply_norm(train_df, union_features, nm["mu"], nm["sigma"])
        val_df_n = apply_norm(val_df, union_features, nm["mu"], nm["sigma"])

        ds_train = MultiLabelTabularDataset(train_df_n, union_features, info["label_cols"])
        ds_val = MultiLabelTabularDataset(val_df_n, union_features, info["label_cols"])

        loaders_train[task_name] = DataLoader(ds_train, batch_size=BATCH, shuffle=True, collate_fn=collate_with_raw)
        loaders_val[task_name] = DataLoader(ds_val, batch_size=BATCH, shuffle=False, collate_fn=collate_with_raw)
        train_sizes[task_name] = len(train_df_n)

        print(f"[MetaIRL v7.4] {task_name} train={len(train_df_n)} val={len(val_df_n)} dropped_missing={missing_stats['rows_dropped_any_label_missing']}")

    model = MTLMetaIRLTransformer(
        input_dim=len(union_features),
        task_label_class_counts=task_label_class_counts,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.15,
        proto_alpha=0.35,
        proto_temperature=1.0,
        detach_probs_for_reward=True,
        enable_meta=enable_meta,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_score = -1e9
    best_state = None
    bad_epochs = 0
    history_rows = []

    print(f"[MetaIRL v7.4] Training starts... exp={exp_name} seed={seed}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0

        for task_name in task_sequence:
            loader = loaders_train[task_name]
            label_cols = tasks_meta[task_name]["label_cols"]
            criteria_dict = criteria_by_task[task_name]

            for X, ys, raws in loader:
                X = X.to(DEVICE)
                ys = [y.to(DEVICE) for y in ys]
                support = None if exp_name == "no_meta" else build_support(X, ys, label_cols, max_per_class=2)

                logits_dict, reward_dict = model(X, task_name, support=support)

                ce_loss = 0.0
                reward_loss = 0.0
                for lbl, y in zip(label_cols, ys):
                    logits = logits_dict[lbl]
                    ce_loss = ce_loss + criteria_dict[lbl](logits, y)
                    probs = torch.softmax(logits.detach(), dim=-1)
                    pred_idx = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
                    pred_names = [tasks_meta[task_name]["class_names"][lbl][i] for i in pred_idx]
                    tgt = [expert_consistency_score(task_name, lbl, raw, pred_name) for raw, pred_name in zip(raws, pred_names)]
                    reward_targets = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)
                    reward_loss = reward_loss + nn.functional.mse_loss(reward_dict[lbl], reward_targets)

                log_var = model.log_vars[task_name]
                task_loss = torch.exp(-log_var) * ce_loss + log_var + reward_weight * reward_loss
                if torch.isnan(task_loss):
                    print(f"[WARN] NaN loss at task {task_name}, skip batch")
                    continue

                optimizer.zero_grad()
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                train_loss_total += float(task_loss.item())
                train_steps += 1

        val_summary = {}
        cm_summary = {}
        score_accum = 0.0
        score_count = 0
        for task_name in task_sequence:
            metrics, cms = evaluate_task(model, loaders_val[task_name], {"tasks": tasks_meta}, task_name, criteria_by_task[task_name])
            val_summary[task_name] = metrics
            cm_summary[task_name] = cms
            for lbl in tasks_meta[task_name]["label_cols"]:
                score_accum += metrics.get(f"{lbl}_macro_f1", 0.0)
                score_count += 1

        macro_score = score_accum / max(score_count, 1)
        val_loss = float(np.mean([val_summary[t]["loss"] for t in val_summary])) if val_summary else 0.0
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(train_steps, 1),
            "val_loss": val_loss,
            "mean_macro_f1": macro_score,
        }
        history_rows.append(row)
        print(f"[Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} mean_macro_f1={row['mean_macro_f1']:.4f}")

        if macro_score > best_score:
            best_score = macro_score
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_meta = {
                "union_features": union_features,
                "tasks": tasks_meta,
                "norm_meta": norm_meta,
                "history": history_rows,
                "best_epoch": epoch,
                "best_mean_macro_f1": float(best_score),
                "train_sizes": train_sizes,
                "cleaning_log": cleaning_log,
                "version": "MetaIRL_v7.4",
                "exp_name": exp_name,
                "seed": seed,
                "enable_meta": enable_meta,
                "reward_weight": reward_weight,
            }
            ckpt = {"model_state_dict": best_state, "meta": best_meta}
            save_path = run_dir / "best_model.pth"
            torch.save(ckpt, save_path)
            print(f"[MetaIRL v7.4] Saved best checkpoint → {save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[MetaIRL v7.4] Early stopping at epoch {epoch}")
                break

    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "training_history.csv"
    hist_df.to_csv(hist_path, index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_summary = {}
    confusion = {}
    for task_name in task_sequence:
        metrics, cms = evaluate_task(model, loaders_val[task_name], {"tasks": tasks_meta}, task_name, criteria_by_task[task_name])
        eval_summary[task_name] = metrics
        confusion[task_name] = cms

    save_json({
        "history": history_rows,
        "eval_summary": eval_summary,
        "confusion_matrices": confusion,
        "cleaning_log": cleaning_log,
    }, run_dir / "eval_summary.json")

    print("[MetaIRL v7.4] Training finished.")
    print(f"[MetaIRL v7.4] Best macro-F1 = {best_score:.4f}")
    return {
        "exp_name": exp_name,
        "seed": seed,
        "best_mean_macro_f1": float(best_score),
        "run_dir": str(run_dir),
    }


def aggregate_results(out_root: str):
    root = Path(out_root)
    rows = []
    for run_dir in sorted(root.glob("*_seed_*")):
        summary_path = run_dir / "eval_summary.json"
        if not summary_path.exists():
            continue
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        exp_name = run_dir.name.split("_seed_")[0]
        seed = int(run_dir.name.split("_seed_")[1])
        row = {"exp_name": exp_name, "seed": seed}
        for task_name, metrics in obj["eval_summary"].items():
            for k, v in metrics.items():
                row[f"{task_name}__{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return None
    df.to_csv(root / "all_runs_metrics.csv", index=False)

    grouped_rows = []
    numeric_cols = [c for c in df.columns if c not in ["exp_name", "seed"]]
    for exp_name, sub in df.groupby("exp_name"):
        g = {"exp_name": exp_name, "n_runs": int(len(sub))}
        for c in numeric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            g[f"{c}__mean"] = float(vals.mean()) if vals.notna().any() else None
            g[f"{c}__std"] = float(vals.std(ddof=0)) if vals.notna().any() else None
        grouped_rows.append(g)
    gdf = pd.DataFrame(grouped_rows)
    gdf.to_csv(root / "summary.csv", index=False)
    save_json(grouped_rows, root / "summary.json")
    return gdf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--results_dir", type=str, default="results_v74")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--exp", type=str, default=None, choices=["full", "no_meta", "no_irl", "single_task"])
    p.add_argument("--experiments", type=str, default="full,no_meta,no_irl,single_task")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.exp is not None:
        experiments = [args.exp]
    else:
        experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    all_runs = []
    for exp_name in experiments:
        for seed in seeds:
            run_info = run_experiment(args.data_dir, args.results_dir, seed, exp_name)
            all_runs.append(run_info)

    save_json(all_runs, Path(args.results_dir) / "run_manifest.json")
    aggregate_results(args.results_dir)
    print(f"[MetaIRL v7.4] All runs finished. Results → {Path(args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
