from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from checkpoint_utils_v74 import DEFAULT_PRIMARY_CHECKPOINT, resolve_checkpoint_paths
from clinical_rules_v74 import has_hard_rule_warning

try:
    import torch
    from mtl_meta_irl_transformer import MTLMetaIRLTransformer
except ModuleNotFoundError:
    torch = None
    MTLMetaIRLTransformer = None


DEFAULT_CHECKPOINT = DEFAULT_PRIMARY_CHECKPOINT

def resolve_torch_device(device_choice: str = "cpu"):
    choice = str(device_choice or "cpu").strip().lower()
    if torch is None:
        if choice.startswith("cuda"):
            raise ModuleNotFoundError("torch is required for CUDA model profiling")
        raise ModuleNotFoundError("torch is required for neural checkpoint profiling")
    if choice in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[model_profile_v74] CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {device_choice} was requested, but CUDA is not available.")
        return torch.device(choice)
    raise ValueError(f"Unsupported --device value: {device_choice}. Use auto, cpu, cuda, or cuda:<index>.")



def load_model(checkpoint_path: str, device: torch.device):
    if torch is None:
        raise ModuleNotFoundError("torch is required for neural checkpoint profiling")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    meta = ckpt["meta"]
    model_config = meta.get("model_config", {})
    model = MTLMetaIRLTransformer(
        input_dim=len(meta["union_features"]),
        task_label_class_counts={task: info["num_classes"] for task, info in meta["tasks"].items()},
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 4)),
        dropout=float(model_config.get("dropout", 0.15)),
        proto_alpha=float(model_config.get("proto_alpha", 0.35)),
        proto_temperature=float(model_config.get("proto_temperature", 1.0)),
        enable_meta=bool(meta.get("enable_meta", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta


def parameter_summary(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
    }


def measure_latency(model, input_dim: int, task_name: str, batch_size: int, warmup: int, repeats: int, device: torch.device):
    x = torch.randn(batch_size, input_dim, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, task_name, support=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x, task_name, support=None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
    mean_ms = 1000.0 * sum(elapsed) / max(len(elapsed), 1)
    return {
        "latency_mean_ms": float(mean_ms),
        "latency_per_sample_ms": float(mean_ms / batch_size),
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
    }


def parse_csv_list(text: str):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def infer_model_name(checkpoint: Path) -> str:
    if checkpoint.parent.name.endswith("_seed_0") or "_seed_" in checkpoint.parent.name:
        return f"{checkpoint.parent.parent.name}/{checkpoint.parent.name}"
    return checkpoint.parent.name or checkpoint.stem


def parse_checkpoint_identity(checkpoint: Path) -> dict:
    exp_seed = checkpoint.parent.name
    match = re.match(r"(?P<exp>.+)_seed_(?P<seed>\d+)$", exp_seed)
    return {
        "config_name": checkpoint.parent.parent.name if checkpoint.parent.parent != checkpoint.parent else None,
        "exp_name": match.group("exp") if match else exp_seed,
        "seed": int(match.group("seed")) if match else None,
    }


def collect_checkpoint_paths(args) -> list[Path]:
    checkpoint = None if args.skip_default_checkpoint else args.checkpoint
    glob_patterns = []
    if getattr(args, "profile_all_default", False):
        glob_patterns.append("results_v74/five_runs/**/best_model.pth")
    if args.checkpoint_glob:
        glob_patterns.append(args.checkpoint_glob)
    try:
        return resolve_checkpoint_paths(
            checkpoint=checkpoint,
            checkpoints=args.checkpoints,
            checkpoint_glob=glob_patterns,
            default_checkpoint=DEFAULT_CHECKPOINT,
            search_defaults=not args.skip_default_checkpoint,
        )
    except FileNotFoundError:
        if args.skip_default_checkpoint and (args.hybrid_predictions or args.include_ml_baselines):
            print("[WARN] No neural checkpoint resolved; continuing with hybrid/ML profile inputs only.")
            return []
        raise


def profile_checkpoint(checkpoint_path: Path, args, model_name: str = None):
    checkpoint = Path(checkpoint_path)
    identity = parse_checkpoint_identity(checkpoint)
    if torch is None:
        return pd.DataFrame([{
            "profile_type": "neural_checkpoint_skipped",
            "model_name": model_name or infer_model_name(checkpoint),
            "checkpoint": str(checkpoint),
            **identity,
            "skip_reason": "torch_not_installed",
        }]), {}
    device = resolve_torch_device(args.device)
    setattr(args, "resolved_device", str(device))
    print(f"[model_profile_v74] Requested device={args.device}; resolved neural device={device}")
    model, meta = load_model(str(checkpoint), device)
    param = parameter_summary(model)
    input_dim = len(meta["union_features"])
    checkpoint_size_mb = checkpoint.stat().st_size / (1024 * 1024)
    model_config = meta.get("model_config", {})
    missing_aug = meta.get("missingness_augmentation", {})

    rows = []
    for task_name in meta["tasks"].keys():
        for batch_size in args.batch_sizes:
            latency = measure_latency(
                model,
                input_dim,
                task_name,
                int(batch_size),
                args.warmup,
                args.repeats,
                device,
            )
            rows.append({
                "profile_type": "neural_checkpoint",
                "model_name": model_name or infer_model_name(checkpoint),
                "checkpoint": str(checkpoint),
                **identity,
                "device": str(device),
                "task": task_name,
                "input_dim": int(input_dim),
                "checkpoint_size_mb": float(checkpoint_size_mb),
                "model_d_model": int(model_config.get("d_model", 128)),
                "model_num_layers": int(model_config.get("num_layers", 4)),
                "missing_aug_p": float(missing_aug.get("p", 0.0) or 0.0),
                **param,
                **latency,
            })
    return pd.DataFrame(rows), meta


def profile_hybrid_predictions(path: Path, args) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hybrid prediction file not found: {path}")
    df = pd.read_csv(path, usecols=lambda col: col in {
        "task",
        "baseline_covered",
        "complete_for_rule",
        "rule_confidence",
        "warning_reasons",
        "pred_label",
        "forced_pred_label",
    })
    if df.empty:
        return pd.DataFrame()
    covered = df["baseline_covered"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    if "complete_for_rule" in df.columns:
        complete_raw = df["complete_for_rule"]
        complete = complete_raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        complete = complete.mask(complete_raw.isna(), covered)
    else:
        complete = covered
    covered = covered.to_numpy()
    complete = complete.astype(bool).to_numpy()
    rule_conf = pd.to_numeric(df["rule_confidence"], errors="coerce").fillna(0.0).to_numpy()
    if "warning_reasons" in df.columns:
        hard_warning = df["warning_reasons"].fillna("").astype(str).map(has_hard_rule_warning).to_numpy(dtype=bool)
    else:
        hard_warning = np.zeros(len(df), dtype=bool)
    model_pred = df["pred_label"].astype(str).to_numpy()
    rule_pred_series = df["forced_pred_label"].fillna("").astype(str).str.strip()
    rule_label_available = (
        rule_pred_series.ne("")
        & rule_pred_series.str.lower().ne("none")
        & rule_pred_series.ne("INSUFFICIENT_EVIDENCE")
    ).to_numpy()
    rule_pred = rule_pred_series.to_numpy()
    rows = []
    for task_name, sub_idx in df.groupby("task", dropna=False).indices.items():
        idx = np.array(list(sub_idx), dtype=int)
        elapsed = []
        for _ in range(args.warmup):
            use_rule = (
                rule_label_available[idx]
                & covered[idx]
                & complete[idx]
                & (rule_conf[idx] >= args.rule_confidence_threshold)
                & (~hard_warning[idx])
            )
            np.where(use_rule, rule_pred[idx], model_pred[idx])
        for _ in range(args.repeats):
            start = time.perf_counter()
            use_rule = (
                rule_label_available[idx]
                & covered[idx]
                & complete[idx]
                & (rule_conf[idx] >= args.rule_confidence_threshold)
                & (~hard_warning[idx])
            )
            np.where(use_rule, rule_pred[idx], model_pred[idx])
            elapsed.append(time.perf_counter() - start)
        mean_ms = 1000.0 * sum(elapsed) / max(len(elapsed), 1)
        rows.append({
            "profile_type": "hybrid_rule_first_decision",
            "model_name": "hybrid_rule_first",
            "checkpoint": "",
            "device": "cpu",
            "task": str(task_name),
            "input_dim": None,
            "checkpoint_size_mb": 0.0,
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "latency_mean_ms": float(mean_ms),
            "latency_per_sample_ms": float(mean_ms / max(len(idx), 1)),
            "batch_size": int(len(idx)),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "rule_confidence_threshold": float(args.rule_confidence_threshold),
            "hybrid_rule_rate": float(np.mean(use_rule)) if len(use_rule) else 0.0,
            "hard_warning_block_rate": float(np.mean(rule_label_available[idx] & hard_warning[idx])) if len(idx) else 0.0,
        })
    return pd.DataFrame(rows)


def measure_predict_latency(estimator, x_values: np.ndarray, batch_size: int, warmup: int, repeats: int):
    if len(x_values) == 0:
        return None
    batch = x_values[: min(batch_size, len(x_values))]
    for _ in range(warmup):
        estimator.predict(batch)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        estimator.predict(batch)
        elapsed.append(time.perf_counter() - start)
    mean_ms = 1000.0 * sum(elapsed) / max(len(elapsed), 1)
    return {
        "latency_mean_ms": float(mean_ms),
        "latency_per_sample_ms": float(mean_ms / max(len(batch), 1)),
        "batch_size": int(len(batch)),
        "warmup": int(warmup),
        "repeats": int(repeats),
    }


def profile_ml_baselines(args) -> pd.DataFrame:
    from ml_baselines_v74 import (
        TASK_INFO,
        build_feature_sets,
        build_models,
        fit_estimator,
        prepare_split,
    )

    raw_dfs, feature_sets = build_feature_sets(Path(args.data_dir), args.feature_mode)
    requested = parse_csv_list(args.ml_models)
    models, skipped = build_models(args.ml_seed, args.quick_ml, requested)
    rows = [
        {
            "profile_type": "classical_ml_skipped",
            "model_name": item.get("model"),
            "task": "ALL",
            "skip_reason": item.get("reason"),
        }
        for item in skipped
    ]
    for task_name in TASK_INFO:
        x_train, y_train, _x_val, y_val, meta = prepare_split(
            raw_dfs[task_name],
            task_name,
            feature_sets[task_name],
            args.ml_seed,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        for model_name, model_spec in models.items():
            estimator = model_spec["estimator"]
            fit_estimator(
                estimator,
                x_train,
                y_train,
                sample_weight=sample_weight if model_spec["sample_weight"] else None,
            )
            for batch_size in args.batch_sizes:
                latency = measure_predict_latency(
                    estimator,
                    _x_val,
                    int(batch_size),
                    args.warmup,
                    args.repeats,
                )
                if latency is None:
                    continue
                rows.append({
                    "profile_type": "classical_ml_predict",
                    "model_name": model_name,
                    "checkpoint": "",
                    "device": "cpu",
                    "task": task_name,
                    "input_dim": int(meta["feature_count"]),
                    "checkpoint_size_mb": None,
                    "parameter_count": None,
                    "trainable_parameter_count": None,
                    "train_n": int(meta["train_n"]),
                    "val_n": int(meta["val_n"]),
                    "feature_mode": args.feature_mode,
                    "quick": bool(args.quick_ml),
                    **latency,
                })
    return pd.DataFrame(rows)


def infer_model_family(row: pd.Series) -> str:
    profile_type = str(row.get("profile_type", "")).lower()
    model_name = str(row.get("model_name", "")).lower()
    d_model = pd.to_numeric(pd.Series([row.get("model_d_model")]), errors="coerce").iloc[0]
    num_layers = pd.to_numeric(pd.Series([row.get("model_num_layers")]), errors="coerce").iloc[0]

    if "hybrid" in profile_type or "hybrid" in model_name:
        return "hybrid_rule_first"
    if any(name in model_name for name in ["hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]):
        return "gbdt_family"
    if "random_forest" in model_name:
        return "random_forest"
    if "logistic" in model_name:
        return "logistic_regression"
    if "mlp" in model_name:
        return "mlp"
    if pd.notna(d_model) and pd.notna(num_layers):
        if int(d_model) <= 32 or "tiny" in model_name:
            return "tiny_transformer"
        if int(d_model) <= 64 or "small" in model_name:
            return "small_transformer"
        return "transformer"
    return "other"


def normalize_merge_key_columns(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in key_cols:
        if col not in out.columns:
            continue
        if col == "seed":
            seed_values = pd.to_numeric(out[col], errors="coerce").astype("Int64").astype(str)
            out[col] = seed_values.replace("<NA>", "")
        else:
            out[col] = out[col].where(out[col].notna(), "").astype(str)
    return out


def build_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["model_family"] = out.apply(infer_model_family, axis=1)
    group_candidates = [
        "profile_type",
        "model_family",
        "model_name",
        "config_name",
        "exp_name",
        "seed",
        "checkpoint",
        "device",
        "task",
    ]
    for col in group_candidates:
        if col not in out.columns:
            continue
        if col == "seed":
            seed_values = pd.to_numeric(out[col], errors="coerce").astype("Int64").astype(str)
            out[col] = seed_values.replace("<NA>", "")
        else:
            out[col] = out[col].where(out[col].notna(), "").astype(str)
    for col in [
        "checkpoint_size_mb",
        "parameter_count",
        "trainable_parameter_count",
        "latency_mean_ms",
        "latency_per_sample_ms",
        "batch_size",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "parameter_count" in out.columns:
        out["approx_fp32_parameter_memory_mb"] = out["parameter_count"] * 4.0 / (1024 * 1024)

    index_cols = [
        col
        for col in [
            *group_candidates,
        ]
        if col in out.columns
    ]
    first_cols = [
        col
        for col in [
            "input_dim",
            "checkpoint_size_mb",
            "approx_fp32_parameter_memory_mb",
            "parameter_count",
            "trainable_parameter_count",
            "model_d_model",
            "model_num_layers",
            "missing_aug_p",
            "feature_mode",
            "quick",
            "skip_reason",
        ]
        if col in out.columns and col not in index_cols
    ]
    grouped = out.groupby(index_cols, dropna=False)
    summary = grouped[first_cols].first().reset_index() if first_cols else grouped.size().reset_index().drop(columns=[0])

    if "batch_size" in out.columns:
        batch_values = sorted(
            int(value)
            for value in out["batch_size"].dropna().unique().tolist()
            if float(value).is_integer()
        )
        for batch_size in batch_values:
            sub = out[out["batch_size"].eq(batch_size)]
            if sub.empty:
                continue
            latency = (
                sub.groupby(index_cols, dropna=False)["latency_mean_ms"]
                .mean()
                .rename(f"batch{batch_size}_latency_ms")
                .reset_index()
            )
            per_sample = (
                sub.groupby(index_cols, dropna=False)["latency_per_sample_ms"]
                .mean()
                .rename(f"batch{batch_size}_per_sample_ms")
                .reset_index()
            )
            summary = normalize_merge_key_columns(summary, index_cols)
            latency = normalize_merge_key_columns(latency, index_cols)
            per_sample = normalize_merge_key_columns(per_sample, index_cols)
            summary = summary.merge(latency, on=index_cols, how="left")
            summary = summary.merge(per_sample, on=index_cols, how="left")

    sort_cols = [col for col in ["model_family", "model_name", "task"] if col in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Profile v7.4 checkpoint size, parameter count, and inference latency.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-default-checkpoint", action="store_true")
    parser.add_argument("--checkpoints", default="", help="Comma-separated additional checkpoints.")
    parser.add_argument("--checkpoint-glob", default="", help="Optional glob for profiling multiple checkpoints.")
    parser.add_argument("--profile-all-default", action="store_true", help="Profile all results_v74/five_runs/**/best_model.pth checkpoints.")
    parser.add_argument("--output-dir", default="results_v74/model_profile")
    parser.add_argument("--device", default="cpu", help="Neural checkpoint profiling device: cpu, auto, cuda, or cuda:<index>. Default cpu preserves edge/deployment latency meaning.")
    parser.add_argument("--batch-sizes", default="1,32,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--hybrid-predictions", default="", help="Optional hybrid_predictions.csv for rule-first latency.")
    parser.add_argument("--rule-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--include-ml-baselines", action="store_true")
    parser.add_argument("--ml-models", default="random_forest,hist_gradient_boosting")
    parser.add_argument("--ml-seed", type=int, default=0)
    parser.add_argument("--quick-ml", action="store_true")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--feature-mode", default="union", choices=["task", "union"])
    args = parser.parse_args()
    args.batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    frames = []
    metas = []
    for checkpoint in collect_checkpoint_paths(args):
        checkpoint_df, meta = profile_checkpoint(checkpoint, args)
        frames.append(checkpoint_df)
        metas.append(meta)
    if args.hybrid_predictions:
        frames.append(profile_hybrid_predictions(Path(args.hybrid_predictions), args))
    if args.include_ml_baselines:
        frames.append(profile_ml_baselines(args))
    usable_frames = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        cleaned = frame.dropna(axis=1, how="all")
        if cleaned.shape[1] == 0:
            continue
        usable_frames.append(cleaned)
    df = pd.concat(usable_frames, ignore_index=True) if usable_frames else pd.DataFrame()
    summary_df = build_profile_summary(df)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_profile.csv"
    summary_csv_path = out_dir / "model_profile_summary.csv"
    deployment_csv_path = out_dir / "deployment_profile.csv"
    manifest_path = out_dir / "model_profile_manifest.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    df.to_csv(deployment_csv_path, index=False, encoding="utf-8-sig")
    feature_count = None
    for meta in metas:
        if isinstance(meta, dict) and "union_features" in meta:
            feature_count = len(meta["union_features"])
            break
    manifest = {
        "checkpoints": [str(path) for path in collect_checkpoint_paths(args)],
        "device": args.device,
        "requested_device": args.device,
        "resolved_device": getattr(args, "resolved_device", args.device),
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "hybrid_predictions": args.hybrid_predictions,
        "include_ml_baselines": bool(args.include_ml_baselines),
        "feature_count": feature_count,
        "outputs": {
            "profile_csv": str(csv_path),
            "summary_csv": str(summary_csv_path),
            "deployment_profile_csv": str(deployment_csv_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(df.to_string(index=False))
    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    print(f"saved={csv_path}")
    print(f"saved_summary={summary_csv_path}")


if __name__ == "__main__":
    main()
