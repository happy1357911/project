import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from checkpoint_utils_v74 import DEFAULT_PRIMARY_CHECKPOINT, resolve_checkpoint_paths
from clinical_rules_v74 import (
    MISSING_TOKENS,
    TASK2_ABG_FREQS,
    TASK2_AC_NR_LIMITS_DB,
    TASK2_BC_NR_LIMITS_DB,
    TASK2_EAR_LEVEL_AC_FREQS,
    TASK3_NAME_MAP,
    TASK3_NP_EXTREME_VALUES,
    clean_value_zero as clean_value,
    infer_task2_hearing_type_from_rule,
    is_nr_value,
    mean_present,
    parse_numeric_value,
    score_rule_decision,
    score_task3_expert_consistency,
    hz_from_col,
)
from mtl_meta_irl_transformer import MTLMetaIRLTransformer
from preprocessing_v74 import (
    INSUFFICIENT_EVIDENCE_LABEL,
    build_model_row as shared_build_model_row,
    clinical_warning_summary,
)


st.set_page_config(page_title="Audiologist-Guided Hearing Decision Support v7.4", layout="wide")

SUPPORTED_CC = {(5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (8, 9)}
DEFAULT_CHECKPOINT = DEFAULT_PRIMARY_CHECKPOINT
DEFAULT_RULE_CONFIDENCE_THRESHOLD = 0.8


def safe_pick_device():
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) not in SUPPORTED_CC:
                st.warning(f"CUDA sm_{major}{minor} is not supported here; using CPU.")
                return torch.device("cpu")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


DEVICE = safe_pick_device()


def checkpoint_candidates(default_path=DEFAULT_CHECKPOINT):
    candidates = resolve_checkpoint_paths(
        checkpoint=default_path,
        checkpoint_glob=[
            "results_v74*/five_runs/**/best_model.pth",
            "all/results_v74*/five_runs/**/best_model.pth",
            "results_v74/*_seed_*/best_model.pth",
            "checkpoints/*.pth",
        ],
        default_checkpoint=default_path,
        search_defaults=True,
        require_exists=False,
    )
    legacy_candidates = [
        "results_v74/full_seed_0/best_model.pth",
        "checkpoints/MTL_MetaIRL_v73_best.pth",
        "checkpoints/MTL_MetaIRL_v72_best.pth",
        "checkpoints/MTL_MetaIRL_v71_best.pth",
        "best_model_metaIRL_v71.pth",
    ]
    seen = set()
    out = []
    for p in [*candidates, *legacy_candidates]:
        path = Path(p)
        if path.exists():
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(str(path))
    return out


@st.cache_resource
def load_model(path=DEFAULT_CHECKPOINT):
    if not os.path.exists(path):
        st.error(f"No checkpoint found: {path}")
        return None, None

    ckpt = torch.load(path, map_location="cpu")
    union_features = ckpt["meta"]["union_features"]
    tasks_meta = ckpt["meta"]["tasks"]
    model_config = ckpt["meta"].get("model_config", {})
    model = MTLMetaIRLTransformer(
        input_dim=len(union_features),
        task_label_class_counts={t: v["num_classes"] for t, v in tasks_meta.items()},
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 4)),
        dropout=float(model_config.get("dropout", 0.15)),
        proto_alpha=float(model_config.get("proto_alpha", 0.35)),
        proto_temperature=float(model_config.get("proto_temperature", 1.0)),
        enable_meta=ckpt["meta"].get("enable_meta", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    ckpt["meta"]["checkpoint_path"] = path
    return model, ckpt["meta"]


st.title("Audiologist-Guided Hearing Decision Support v7.4")
st.caption("Ear-level task-conditioned inference")

with st.sidebar:
    st.subheader("Checkpoint")
    candidates = checkpoint_candidates()
    if not candidates:
        st.error("No checkpoint found under results_v74*/five_runs.")
        st.stop()
    checkpoint_path = st.selectbox("Model checkpoint", candidates, index=0)

model, meta = load_model(checkpoint_path)
if model is None:
    st.stop()


def row_has(row, col):
    return col in row and str(row.get(col)).strip().lower() not in MISSING_TOKENS


def row_numeric(row, col, nr_limits=None):
    return parse_numeric_value(row.get(col), hz=hz_from_col(col), nr_limits=nr_limits)


def first_numeric(row, cols, nr_limits=None):
    for col in cols:
        if col in row:
            x = row_numeric(row, col, nr_limits=nr_limits)
            if x is not None:
                return x
    return None


def first_nr_flag(row, cols):
    for col in cols:
        if col in row and is_nr_value(row.get(col)):
            return 1.0
    return 0.0


def flag_value(row, col, fallback=0.0):
    value = row.get(col, fallback)
    try:
        if pd.isna(value) or str(value).strip() == "":
            return float(fallback)
        return 1.0 if float(value) >= 0.5 else 0.0
    except Exception:
        return float(fallback)


def normalize_side(side):
    side = str(side).strip().lower()
    return side if side in {"right", "left"} else "right"


def task3_flags(raw_value):
    raw = str(raw_value).strip()
    numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    is_np = raw.upper() == "NP"
    is_missing = raw.lower() in MISSING_TOKENS or pd.isna(raw_value)
    is_real_zero = bool(pd.notna(numeric) and float(numeric) == 0.0 and not is_missing and not is_np)
    return float(is_real_zero), float(is_missing), float(is_np)


def build_task1_row(row, side):
    out = dict(row)
    out["ear_side"] = side
    for hz in [500, 1000, 2000, 4000]:
        x = first_numeric(row, [f"ac_{hz}Hz", f"{side}_{hz}Hz"], TASK2_AC_NR_LIMITS_DB)
        out[f"ac_{hz}Hz"] = 0.0 if x is None else x
    pta = first_numeric(row, ["ac_PTA", f"{side}_PTA"], TASK2_AC_NR_LIMITS_DB)
    out["ac_PTA"] = 0.0 if pta is None else pta
    return out


def build_task2_row(row, side):
    out = build_task1_row(row, side)
    bc_values = []
    abg_values = []
    abg_censored_values = []

    for hz in TASK2_EAR_LEVEL_AC_FREQS:
        ac_cols = [f"ac_{hz}Hz", f"{side}_{hz}Hz"]
        x = first_numeric(row, ac_cols, TASK2_AC_NR_LIMITS_DB)
        out[f"ac_{hz}Hz"] = 0.0 if x is None else x
        out[f"ac_{hz}Hz_nr"] = flag_value(row, f"ac_{hz}Hz_nr", first_nr_flag(row, ac_cols))

    for hz in TASK2_ABG_FREQS:
        ac_cols = [f"ac_{hz}Hz", f"{side}_{hz}Hz"]
        bc_cols = [f"bc_{hz}Hz", f"bc_{side}_{hz}Hz"]
        ac = first_numeric(row, ac_cols, TASK2_AC_NR_LIMITS_DB)
        bc = first_numeric(row, bc_cols, TASK2_BC_NR_LIMITS_DB)
        bc_flag_col = f"bc_{hz}Hz_missing"
        abg_flag_col = f"abg_{hz}Hz_missing"
        abg_censored_col = f"abg_{hz}Hz_censored"
        ac_nr = flag_value(row, f"ac_{hz}Hz_nr", first_nr_flag(row, ac_cols))
        bc_nr = flag_value(row, f"bc_{hz}Hz_nr", first_nr_flag(row, bc_cols))

        bc_missing = clean_value(row.get(bc_flag_col, 1.0 if bc is None else 0.0)) >= 0.5
        abg = row_numeric(row, f"abg_{hz}Hz")
        if abg is None and ac is not None and bc is not None:
            abg = ac - bc
        abg_missing = clean_value(row.get(abg_flag_col, 1.0 if (ac is None or bc is None) else 0.0)) >= 0.5
        abg_censored = flag_value(row, abg_censored_col, 1.0 if ((ac_nr >= 0.5 or bc_nr >= 0.5) and not abg_missing) else 0.0)

        out[f"bc_{hz}Hz"] = 0.0 if bc is None else bc
        out[f"bc_{hz}Hz_nr"] = bc_nr
        out[bc_flag_col] = float(bc_missing)
        out[f"abg_{hz}Hz"] = 0.0 if abg is None else abg
        out[abg_flag_col] = float(abg_missing)
        out[abg_censored_col] = float(abg_censored)
        if not bc_missing:
            bc_values.append(bc)
        if not abg_missing:
            abg_values.append(abg)
            abg_censored_values.append(abg_censored)

    bc_mean = row_numeric(row, "bc_mean")
    abg_mean = row_numeric(row, "abg_mean")
    out["bc_mean"] = 0.0 if bc_mean is None else bc_mean
    if bc_mean is None:
        out["bc_mean"] = 0.0 if mean_present(bc_values) is None else mean_present(bc_values)
    out["abg_mean"] = 0.0 if abg_mean is None else abg_mean
    if abg_mean is None:
        out["abg_mean"] = 0.0 if mean_present(abg_values) is None else mean_present(abg_values)
    out["abg_mean_censored"] = flag_value(
        row,
        "abg_mean_censored",
        max(abg_censored_values) if abg_censored_values else 0.0,
    )
    return out


def build_task3_row(row, side):
    out = dict(row)
    out["ear_side"] = side
    for suffix, neutral_col in TASK3_NAME_MAP.items():
        legacy_col = f"tymp_{side}_{suffix}"
        raw_value = row.get(neutral_col, row.get(legacy_col, ""))
        real_zero, missing_zero, np_zero = task3_flags(raw_value)
        value = parse_numeric_value(raw_value)
        if np_zero >= 0.5 and neutral_col in TASK3_NP_EXTREME_VALUES:
            value = TASK3_NP_EXTREME_VALUES[neutral_col]
        out[neutral_col] = 0.0 if value is None else value
        for flag, default in [
            ("real_zero", real_zero),
            ("missing_zero", missing_zero),
            ("np_zero", np_zero),
        ]:
            neutral_flag = f"{neutral_col}_{flag}"
            legacy_flag = f"{legacy_col}_{flag}"
            out[neutral_flag] = clean_value(row.get(neutral_flag, row.get(legacy_flag, default)))
    return out


def build_model_row(row, task_name, side):
    return shared_build_model_row(row, task_name, side)


def norm_row(task_name, row_dict):
    feats = meta["union_features"]
    mu = meta["norm_meta"][task_name]["mu"]
    sigma = meta["norm_meta"][task_name]["sigma"]
    out = []
    for f in feats:
        x = clean_value(row_dict.get(f, 0.0))
        out.append((x - mu[f]) / sigma[f])
    return np.array(out, dtype=np.float32)


def expert_consistency_score(task_name, label_col, row, pred_name):
    return float(score_rule_decision(task_name, row, pred_name))


def predict_single(row_dict, task_name, side="right"):
    model_row = build_model_row(row_dict, task_name, side)
    x = norm_row(task_name, model_row)
    x = torch.tensor(x).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_dict, reward_dict = model(x, task_name, support=None)

    results = {}
    label_cols = meta["tasks"][task_name]["label_cols"]
    class_names = meta["tasks"][task_name]["class_names"]
    for lbl in label_cols:
        logits = logits_dict[lbl][0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[lbl][pred_idx]
        reward_pred = float(reward_dict[lbl][0].cpu().item())
        expert = expert_consistency_score(task_name, lbl, model_row, pred_name)
        warning = clinical_warning_summary(task_name, model_row, model_pred=pred_name)
        rule_label = warning.get("forced_rule_label")
        rule_confidence = float(warning.get("rule_confidence") or 0.0)
        baseline_covered = bool(warning.get("baseline_covered", False))
        hybrid_uses_rule = (
            baseline_covered
            and rule_label not in {None, "", INSUFFICIENT_EVIDENCE_LABEL}
            and rule_confidence >= DEFAULT_RULE_CONFIDENCE_THRESHOLD
        )
        hybrid_decision = rule_label if hybrid_uses_rule else pred_name
        results[lbl] = {
            "model_prediction": pred_name,
            "pred": pred_name,
            "probs": probs,
            "reward_pred": reward_pred,
            "expert_consistency": expert,
            "rule_label": rule_label,
            "abstain_rule_label": warning.get("abstain_rule_label"),
            "hybrid_decision": hybrid_decision,
            "hybrid_source": "rule" if hybrid_uses_rule else "model",
            "hybrid_uses_rule": hybrid_uses_rule,
            "baseline_covered": baseline_covered,
            "evidence_status": warning.get("evidence_status"),
            "rule_confidence": rule_confidence,
            "rule_model_conflict": warning.get("rule_model_conflict"),
            "warning_reasons": warning.get("warning_reasons"),
            "class_names": class_names[lbl],
        }
    return results


def batch_sides_for_row(row):
    if row_has(row, "ear_side"):
        return [normalize_side(row.get("ear_side"))]
    return ["right", "left"]


def batch_predict(df, task_name):
    outs, errs = [], []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        for side in batch_sides_for_row(row_dict):
            try:
                res = predict_single(row_dict, task_name, side=side)
                row_out = {"index": idx, "ear_side": side}
                for lbl, info in res.items():
                    row_out[f"{lbl}_model_prediction"] = info["model_prediction"]
                    row_out[f"{lbl}_rule_prediction"] = info["rule_label"]
                    row_out[f"{lbl}_hybrid_decision"] = info["hybrid_decision"]
                    row_out[f"{lbl}_hybrid_source"] = info["hybrid_source"]
                    row_out[f"{lbl}_pred"] = info["hybrid_decision"]
                    row_out[f"{lbl}_reward"] = info["reward_pred"]
                    row_out[f"{lbl}_expert_consistency"] = info["expert_consistency"]
                    row_out[f"{lbl}_rule_label"] = info["rule_label"]
                    row_out[f"{lbl}_abstain_rule_label"] = info["abstain_rule_label"]
                    row_out[f"{lbl}_evidence_status"] = info["evidence_status"]
                    row_out[f"{lbl}_rule_confidence"] = info["rule_confidence"]
                    row_out[f"{lbl}_rule_model_conflict"] = info["rule_model_conflict"]
                    row_out[f"{lbl}_warning_reasons"] = info["warning_reasons"]
                    for i, cname in enumerate(info["class_names"]):
                        row_out[f"{lbl}_prob_{cname}"] = info["probs"][i]
                outs.append(row_out)
            except Exception as e:
                errs.append({"index": idx, "ear_side": side, "error": str(e)})
    return pd.DataFrame(outs), pd.DataFrame(errs)


task_sampling = meta.get("task_sampling", {})
meta_episode = meta.get("meta_episode", {})
st.info(
    f"Checkpoint: {meta.get('checkpoint_path', 'unknown')} | "
    f"input_dim={len(meta['union_features'])} | "
    f"task_sampling={task_sampling.get('mode', 'unknown')} | "
    f"support_per_class={meta_episode.get('support_per_class', 'unknown')}"
)


def side_num(label, default=0.0):
    return st.number_input(label, value=default, step=0.1)


with st.sidebar:
    st.subheader("Input")
    selected_side = st.radio("Ear side", ["right", "left"], horizontal=True)
    vals = {}
    for side in ["right", "left"]:
        prefix = "R" if side == "right" else "L"
        vals[f"{side}_500Hz"] = side_num(f"{prefix} AC 500Hz")
        vals[f"{side}_1000Hz"] = side_num(f"{prefix} AC 1000Hz")
        vals[f"{side}_2000Hz"] = side_num(f"{prefix} AC 2000Hz")
        vals[f"{side}_4000Hz"] = side_num(f"{prefix} AC 4000Hz")
        vals[f"{side}_PTA"] = side_num(f"{prefix} PTA")
        vals[f"bc_{side}_500Hz"] = side_num(f"{prefix} BC 500Hz")
        vals[f"bc_{side}_1000Hz"] = side_num(f"{prefix} BC 1000Hz")
        vals[f"bc_{side}_2000Hz"] = side_num(f"{prefix} BC 2000Hz")
        vals[f"bc_{side}_4000Hz"] = side_num(f"{prefix} BC 4000Hz")
        vals[f"tymp_{side}_Vea"] = side_num(f"{prefix} tymp Vea")
        vals[f"tymp_{side}_peak_daPa"] = side_num(f"{prefix} tymp peak daPa")
        vals[f"tymp_{side}_peak_mmho"] = side_num(f"{prefix} tymp peak mmho")
        vals[f"tymp_{side}_Width_daPa"] = side_num(f"{prefix} tymp Width daPa")


tabs = st.tabs(["Task1 WHO", "Task2 Type", "Task3 Tymp"])
for task_name, tab in zip(["Task1", "Task2", "Task3"], tabs):
    with tab:
        st.subheader(task_name)
        if st.button(f"Predict {task_name}", key=f"single_{task_name}"):
            out = predict_single(vals, task_name, side=selected_side)
            for lbl, info in out.items():
                st.markdown(f"### {lbl} ({selected_side})")
                st.write(f"Model prediction: **{info['model_prediction']}**")
                st.write(f"Rule prediction: **{info['rule_label']}**")
                st.write(f"Hybrid decision: **{info['hybrid_decision']}** (`{info['hybrid_source']}`)")
                st.write(f"Reward head: `{info['reward_pred']:.3f}`")
                st.write(f"Expert consistency: `{info['expert_consistency']:.3f}`")
                st.write(f"Evidence: `{info['evidence_status']}` | Rule confidence: `{info['rule_confidence']:.3f}`")
                if info["warning_reasons"]:
                    st.warning(f"Warning: {info['warning_reasons']}")
                if info["rule_model_conflict"]:
                    st.error("Model prediction conflicts with the clinical rule output.")
                chart_df = pd.DataFrame({"class": info["class_names"], "probability": info["probs"]})
                st.bar_chart(chart_df.set_index("class"))

        st.write("---")
        upload = st.file_uploader(f"Upload {task_name} CSV", type=["csv"], key=f"file_{task_name}")
        if upload and st.button(f"Batch predict {task_name}", key=f"batch_{task_name}"):
            df = pd.read_csv(upload)
            preds, errors = batch_predict(df, task_name)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"AudiologistGuided_v74_{task_name}_{now}.xlsx"
            with pd.ExcelWriter(out_path) as writer:
                preds.to_excel(writer, sheet_name="predictions", index=False)
                errors.to_excel(writer, sheet_name="errors", index=False)
            st.success(f"Exported: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download Excel", data=f.read(), file_name=out_path)
