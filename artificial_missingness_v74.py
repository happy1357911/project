import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from checkpoint_utils_v74 import DEFAULT_PRIMARY_CHECKPOINT, resolve_checkpoint_paths
from clinical_rules_v74 import TASK2_ABG_FREQS, TASK2_BC_NR_LIMITS_DB, TASK2_AC_NR_LIMITS_DB, clean_value_zero, has_hard_rule_warning
from mtl_meta_irl_transformer import MTLMetaIRLTransformer
from preprocessing_v74 import (
    INSUFFICIENT_EVIDENCE_LABEL,
    load_tabular_data,
    TASK_INFO,
    clean_label_columns,
    clinical_warning_summary,
    pad_and_clean,
    prepare_task_dataframe,
)


DEFAULT_CHECKPOINT = DEFAULT_PRIMARY_CHECKPOINT

def resolve_torch_device(device_choice: str = "auto") -> torch.device:
    choice = str(device_choice or "auto").strip().lower()
    if choice in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[artificial_missingness_v74] CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {device_choice} was requested, but CUDA is not available.")
        return torch.device(choice)
    raise ValueError(f"Unsupported --device value: {device_choice}. Use auto, cpu, cuda, or cuda:<index>.")



SCENARIO_METADATA = {
    "Task1": {
        "none": {
            "scenario_family": "complete_data",
            "removed_evidence": "none",
            "fallback_evidence": "not_applicable",
            "clinical_question": "complete PTA baseline",
        },
        "task1_no_pta": {
            "scenario_family": "pta_missing",
            "removed_evidence": "AC PTA summary",
            "fallback_evidence": "frequency-specific AC thresholds",
            "clinical_question": "robustness when PTA summary is unavailable",
        },
        "task1_no_high_freq": {
            "scenario_family": "high_frequency_ac_missing",
            "removed_evidence": "AC 2000/4000 Hz and PTA summary",
            "fallback_evidence": "low-frequency AC thresholds",
            "clinical_question": "robustness when high-frequency AC evidence is unavailable",
        },
        "task1_no_low_freq": {
            "scenario_family": "low_frequency_ac_missing",
            "removed_evidence": "AC 500/1000 Hz and PTA summary",
            "fallback_evidence": "high-frequency AC thresholds",
            "clinical_question": "robustness when low-frequency AC evidence is unavailable",
        },
        "task1_no_all_ac": {
            "scenario_family": "all_ac_missing",
            "removed_evidence": "all Task1 AC evidence",
            "fallback_evidence": "missingness-imputed feature baseline only",
            "clinical_question": "stress test when Task1 AC evidence is unavailable",
        },
    },
    "Task2": {
        "none": {
            "scenario_family": "complete_data",
            "removed_evidence": "none",
            "fallback_evidence": "AC, BC, ABG, NR/censored flags",
            "clinical_question": "complete air/bone audiometry baseline",
        },
        "task2_ac_only": {
            "scenario_family": "bc_missing",
            "removed_evidence": "BC thresholds and ABG evidence",
            "fallback_evidence": "AC thresholds and AC NR flags",
            "clinical_question": "can hearing type be inferred without bone conduction",
        },
        "task2_no_bc": {
            "scenario_family": "bc_missing",
            "removed_evidence": "all BC thresholds and ABG evidence",
            "fallback_evidence": "AC thresholds and AC NR flags",
            "clinical_question": "robustness when BC is unavailable",
        },
        "task2_partial_bc_500_1000": {
            "scenario_family": "partial_bc_missing",
            "removed_evidence": "BC/ABG at 500 and 1000 Hz",
            "fallback_evidence": "remaining BC/ABG frequencies plus AC pattern",
            "clinical_question": "robustness under partial low-frequency BC missingness",
        },
        "task2_multi_frequency_nr": {
            "scenario_family": "nr_or_censored",
            "removed_evidence": "uncensored AC evidence at 2000 and 4000 Hz",
            "fallback_evidence": "remaining AC/BC/ABG and NR/censored flags",
            "clinical_question": "robustness under multi-frequency no-response values",
        },
        "task2_abg_borderline_shift": {
            "scenario_family": "borderline_abg",
            "removed_evidence": "clear ABG separation",
            "fallback_evidence": "borderline ABG warning plus AC/BC threshold pattern",
            "clinical_question": "robustness when ABG is diagnostically borderline",
        },
    },
    "Task3": {
        "none": {
            "scenario_family": "complete_data",
            "removed_evidence": "none",
            "fallback_evidence": "tympanogram peak, compliance, width, Vea, zero/NP flags",
            "clinical_question": "complete tympanogram baseline",
        },
        "task3_no_peak": {
            "scenario_family": "tymp_peak_missing",
            "removed_evidence": "tympanogram peak pressure and compliance",
            "fallback_evidence": "Vea, width, and missing/NP flags",
            "clinical_question": "robustness without peak-related tympanogram evidence",
        },
        "task3_no_width": {
            "scenario_family": "tymp_width_missing",
            "removed_evidence": "tympanogram width",
            "fallback_evidence": "peak pressure, compliance, Vea, and zero/NP flags",
            "clinical_question": "robustness without tympanogram width",
        },
        "task3_np_like": {
            "scenario_family": "np_like",
            "removed_evidence": "normal measurable peak morphology",
            "fallback_evidence": "NP flags and extreme peak/compliance values",
            "clinical_question": "robustness under no-peak-like tympanogram pattern",
        },
        "task3_no_tymp": {
            "scenario_family": "tymp_missing",
            "removed_evidence": "all tympanogram numeric evidence",
            "fallback_evidence": "missingness flags only; clinical abstention may be appropriate",
            "clinical_question": "robustness when tympanometry is unavailable",
        },
    },
}

SCENARIO_METADATA_COLUMNS = [
    "scenario_family",
    "removed_evidence",
    "fallback_evidence",
    "clinical_question",
    "clinical_missingness_type",
    "expected_decision_policy",
]

SCENARIO_FAMILY_TAXONOMY = {
    "complete_data": {
        "clinical_missingness_type": "complete_evidence",
        "expected_decision_policy": "baseline rule/model comparison",
    },
    "pta_missing": {
        "clinical_missingness_type": "pta_summary_missing",
        "expected_decision_policy": "use frequency-specific AC evidence when available",
    },
    "high_frequency_ac_missing": {
        "clinical_missingness_type": "partial_ac_missing",
        "expected_decision_policy": "model fallback or low-confidence decision if high-frequency evidence is clinically needed",
    },
    "low_frequency_ac_missing": {
        "clinical_missingness_type": "partial_ac_missing",
        "expected_decision_policy": "model fallback or low-confidence decision if low-frequency evidence is clinically needed",
    },
    "all_ac_missing": {
        "clinical_missingness_type": "core_ac_missing",
        "expected_decision_policy": "model fallback; rule evidence is insufficient",
    },
    "bc_missing": {
        "clinical_missingness_type": "core_bc_missing",
        "expected_decision_policy": "model fallback; hearing type rule is not clinically supported without BC/ABG",
    },
    "partial_bc_missing": {
        "clinical_missingness_type": "partial_core_bc_missing",
        "expected_decision_policy": "model fallback or rule-with-warning depending on remaining BC/ABG evidence",
    },
    "nr_or_censored": {
        "clinical_missingness_type": "censored_threshold_value",
        "expected_decision_policy": "allow rule only if evidence is otherwise complete and not directionally inconsistent",
    },
    "borderline_abg": {
        "clinical_missingness_type": "borderline_diagnostic_evidence",
        "expected_decision_policy": "model fallback or rule-with-warning; do not treat as clear ABG",
    },
    "tymp_peak_missing": {
        "clinical_missingness_type": "core_tympanogram_peak_missing",
        "expected_decision_policy": "model fallback; tympanogram type rule is not clinically supported without peak evidence",
    },
    "tymp_width_missing": {
        "clinical_missingness_type": "tympanogram_width_missing",
        "expected_decision_policy": "rule-with-warning or model fallback depending on peak/compliance consistency",
    },
    "np_like": {
        "clinical_missingness_type": "no_peak_like_tympanogram_pattern",
        "expected_decision_policy": "model fallback or uncertainty warning; avoid hard B/C decision from numeric peak alone",
    },
    "tymp_missing": {
        "clinical_missingness_type": "all_tympanogram_missing",
        "expected_decision_policy": "model fallback; clinical rule evidence is unavailable",
    },
}

SCENARIO_FORCE_MODEL_FALLBACK = {
    "Task2": {
        "task2_ac_only",
        "task2_no_bc",
        "task2_partial_bc_500_1000",
        "task2_abg_borderline_shift",
    },
    "Task3": {
        "task3_no_peak",
        "task3_no_width",
        "task3_np_like",
        "task3_no_tymp",
    },
}

ARTIFICIAL_MISSINGNESS_OUTPUT_FILES = [
    "artificial_missingness_summary.csv",
    "artificial_missingness_degradation_summary.csv",
    "artificial_missingness_per_class.csv",
    "evidence_compensation_summary.csv",
    "missingness_hybrid_reason_summary.csv",
    "clinical_missingness_taxonomy.csv",
    "artificial_missingness_predictions.csv",
    "artificial_missingness_manifest.json",
]

PAPER_MISSINGNESS_OUTPUT_FILES = [
    "missingness_degradation_summary.csv",
    "missingness_evidence_compensation_summary.csv",
    "missingness_hybrid_reason_summary.csv",
    "clinical_missingness_taxonomy.csv",
]

PREDICTION_CHUNKSIZE = 100_000

def load_model(checkpoint_path: str, device: torch.device):
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


def task_scenarios(task_name: str) -> list[str]:
    if task_name == "Task1":
        return [
            "none",
            "task1_no_pta",
            "task1_no_high_freq",
            "task1_no_low_freq",
            "task1_no_all_ac",
        ]
    if task_name == "Task2":
        return [
            "none",
            "task2_ac_only",
            "task2_no_bc",
            "task2_partial_bc_500_1000",
            "task2_multi_frequency_nr",
            "task2_abg_borderline_shift",
        ]
    if task_name == "Task3":
        return [
            "none",
            "task3_no_peak",
            "task3_no_width",
            "task3_np_like",
            "task3_no_tymp",
        ]
    return ["none"]


def apply_artificial_missingness(df: pd.DataFrame, task_name: str, scenario: str) -> pd.DataFrame:
    out = df.copy()
    if scenario == "none":
        return out

    # Backward-compatible aliases from the earlier smoke-test script.
    aliases = {
        "task1_mask_pta": "task1_no_pta",
        "task1_mask_high_freq": "task1_no_high_freq",
        "task1_mask_low_freq": "task1_no_low_freq",
        "task1_mask_all_ac": "task1_no_all_ac",
        "task2_drop_all_bc": "task2_no_bc",
        "task2_drop_bc_4000": "task2_partial_bc_4000",
        "task2_mask_ac_4000": "task2_ac_4000_missing",
        "task2_simulate_ac_4000_nr": "task2_multi_frequency_nr",
        "task2_simulate_bc_4000_nr": "task2_bc_4000_nr",
        "task3_drop_tymp_all": "task3_no_tymp",
        "task3_drop_tymp_peak": "task3_no_peak",
    }
    scenario = aliases.get(scenario, scenario)

    if task_name == "Task1":
        if scenario == "task1_no_pta":
            cols = ["ac_PTA"]
        elif scenario == "task1_no_high_freq":
            cols = ["ac_2000Hz", "ac_4000Hz", "ac_PTA"]
        elif scenario == "task1_no_low_freq":
            cols = ["ac_500Hz", "ac_1000Hz", "ac_PTA"]
        elif scenario == "task1_no_all_ac":
            cols = list(TASK_INFO["Task1"]["feature_cols"])
        else:
            cols = []
        for col in cols:
            out[col] = np.nan

    if task_name == "Task2":
        if scenario in {"task2_ac_only", "task2_no_bc"}:
            freqs = TASK2_ABG_FREQS
        elif scenario == "task2_partial_bc_500_1000":
            freqs = [500, 1000]
        elif scenario == "task2_partial_bc_4000":
            freqs = [4000]
        else:
            freqs = []

        for hz in freqs:
            out[f"bc_{hz}Hz"] = np.nan
            out[f"bc_{hz}Hz_nr"] = 0.0
            out[f"bc_{hz}Hz_missing"] = 1.0
            out[f"abg_{hz}Hz"] = np.nan
            out[f"abg_{hz}Hz_missing"] = 1.0
            out[f"abg_{hz}Hz_censored"] = 0.0

        if scenario == "task2_ac_only":
            for hz in TASK2_ABG_FREQS:
                out[f"abg_{hz}Hz"] = np.nan
                out[f"abg_{hz}Hz_missing"] = 1.0
                out[f"abg_{hz}Hz_censored"] = 0.0

        if scenario == "task2_ac_4000_missing":
            out["ac_4000Hz"] = np.nan
            out["ac_4000Hz_nr"] = 0.0
            out["abg_4000Hz"] = np.nan
            out["abg_4000Hz_missing"] = 1.0
            out["abg_4000Hz_censored"] = 0.0

        if scenario == "task2_multi_frequency_nr":
            for hz in (2000, 4000, 6000, 8000):
                out[f"ac_{hz}Hz"] = TASK2_AC_NR_LIMITS_DB.get(hz, 120.0)
                out[f"ac_{hz}Hz_nr"] = 1.0
                if hz in TASK2_ABG_FREQS:
                    bc = pd.to_numeric(out.get(f"bc_{hz}Hz", np.nan), errors="coerce")
                    out[f"abg_{hz}Hz"] = out[f"ac_{hz}Hz"] - bc
                    out[f"abg_{hz}Hz_missing"] = bc.isna().astype(float)
                    out[f"abg_{hz}Hz_censored"] = (~bc.isna()).astype(float)

        if scenario == "task2_bc_4000_nr":
            out["bc_4000Hz"] = TASK2_BC_NR_LIMITS_DB.get(4000, 75.0)
            out["bc_4000Hz_nr"] = 1.0
            out["bc_4000Hz_missing"] = 0.0
            ac = pd.to_numeric(out.get("ac_4000Hz", np.nan), errors="coerce")
            out["abg_4000Hz"] = ac - out["bc_4000Hz"]
            out["abg_4000Hz_missing"] = ac.isna().astype(float)
            out["abg_4000Hz_censored"] = (~ac.isna()).astype(float)

        if scenario == "task2_abg_borderline_shift":
            for hz in TASK2_ABG_FREQS:
                out[f"bc_{hz}Hz_missing"] = 0.0
                out[f"bc_{hz}Hz_nr"] = 0.0
                ac = pd.to_numeric(out.get(f"ac_{hz}Hz", np.nan), errors="coerce")
                out[f"bc_{hz}Hz"] = ac - 10.0
                out[f"abg_{hz}Hz"] = 10.0
                out[f"abg_{hz}Hz_missing"] = ac.isna().astype(float)
                out[f"abg_{hz}Hz_censored"] = 0.0

    if task_name == "Task3":
        if scenario == "task3_no_tymp":
            cols = ["tymp_Vea", "tymp_peak_daPa", "tymp_peak_mmho", "tymp_Width_daPa"]
        elif scenario == "task3_no_peak":
            cols = ["tymp_peak_daPa", "tymp_peak_mmho"]
        elif scenario == "task3_no_width":
            cols = ["tymp_Width_daPa"]
        else:
            cols = []
        for col in cols:
            out[col] = np.nan
            out[f"{col}_real_zero"] = 0.0
            out[f"{col}_missing_zero"] = 1.0
            out[f"{col}_np_zero"] = 0.0

        if scenario == "task3_np_like":
            np_values = {
                "tymp_peak_daPa": -999.0,
                "tymp_peak_mmho": -1.0,
                "tymp_Vea": 0.0,
                "tymp_Width_daPa": 0.0,
            }
            for col, value in np_values.items():
                out[col] = value
                out[f"{col}_real_zero"] = 0.0
                out[f"{col}_missing_zero"] = 0.0
                out[f"{col}_np_zero"] = 1.0

    return out


def normalize_features(df: pd.DataFrame, task_name: str, meta: dict) -> np.ndarray:
    union_features = meta["union_features"]
    norm = meta["norm_meta"][task_name]
    df = pad_and_clean(df, union_features)
    values = []
    for feature in union_features:
        x = pd.to_numeric(df[feature], errors="coerce").fillna(0.0).astype(float)
        mu = float(norm["mu"][feature])
        sigma = float(norm["sigma"][feature])
        values.append(((x - mu) / sigma).to_numpy(dtype=np.float32))
    return np.stack(values, axis=1)


def predict_task(model, meta: dict, df: pd.DataFrame, task_name: str, batch_size: int, device: torch.device):
    label_col = TASK_INFO[task_name]["label_cols"][0]
    class_names = meta["tasks"][task_name]["class_names"][label_col]
    X = normalize_features(df, task_name, meta)
    preds = []
    confidences = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits_dict, _ = model(xb, task_name, support=None)
            probs = torch.softmax(logits_dict[label_col], dim=-1).cpu().numpy()
        idx = probs.argmax(axis=1)
        preds.extend([class_names[int(i)] for i in idx])
        confidences.extend(probs[np.arange(len(idx)), idx].astype(float).tolist())
    return preds, confidences


def scenario_metadata(task_name: str, scenario: str) -> dict:
    metadata = dict(SCENARIO_METADATA.get(task_name, {}).get(scenario, {}))
    family = metadata.get("scenario_family", "unknown")
    taxonomy = SCENARIO_FAMILY_TAXONOMY.get(family, {})
    metadata.update(taxonomy)
    return {col: metadata.get(col, "unknown") for col in SCENARIO_METADATA_COLUMNS}


def build_clinical_missingness_taxonomy(tasks: list[str] | None = None) -> pd.DataFrame:
    selected_tasks = tasks or list(SCENARIO_METADATA.keys())
    rows = []
    for task_name in selected_tasks:
        for scenario in task_scenarios(task_name):
            context = scenario_metadata(task_name, scenario)
            force_model = scenario in SCENARIO_FORCE_MODEL_FALLBACK.get(task_name, set())
            rows.append({
                "task": task_name,
                "scenario": scenario,
                **context,
                "scenario_forced_model_fallback": bool(force_model),
                "rule_first_allowed_by_scenario": bool(not force_model and scenario == "none"),
            })
    columns = [
        "task",
        "scenario",
        *SCENARIO_METADATA_COLUMNS,
        "scenario_forced_model_fallback",
        "rule_first_allowed_by_scenario",
    ]
    return pd.DataFrame(rows, columns=columns)


def class_metric_rows(base_info: dict, task_name: str, scenario: str, strategy: str, y_true, y_pred) -> list[dict]:
    y_true = [str(value) for value in y_true]
    y_pred = [str(value) for value in y_pred]
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return []
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    rows = []
    context = scenario_metadata(task_name, scenario)
    for idx, label in enumerate(labels):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fn - fp)
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else np.nan
        rows.append({
            **base_info,
            **context,
            "scenario": scenario,
            "strategy": strategy,
            "task": task_name,
            "class_label": label,
            "support": int(tp + fn),
            "predicted_count": int(tp + fp),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": float(sensitivity) if pd.notna(sensitivity) else np.nan,
            "specificity": float(specificity) if pd.notna(specificity) else np.nan,
            "precision": float(precision) if pd.notna(precision) else np.nan,
            "f1": float(f1) if pd.notna(f1) else np.nan,
        })
    return rows


def macro_sensitivity_specificity(y_true, y_pred) -> tuple[float, float]:
    rows = class_metric_rows({}, "", "", "", y_true, y_pred)
    if not rows:
        return np.nan, np.nan
    sensitivity = pd.to_numeric(pd.Series([row["sensitivity"] for row in rows]), errors="coerce").mean()
    specificity = pd.to_numeric(pd.Series([row["specificity"] for row in rows]), errors="coerce").mean()
    return float(sensitivity), float(specificity)


def metric_dict(task_name: str, y_true, y_pred):
    y_true = [str(value) for value in y_true]
    y_pred = [str(value) for value in y_pred]
    labels = sorted(set(y_true) | set(y_pred))
    macro_sensitivity, macro_specificity = macro_sensitivity_specificity(y_true, y_pred)
    return {
        "task": task_name,
        "n": int(len(y_true)),
        "labels": ",".join(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
    }


def add_complete_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "scenario" not in summary.columns:
        return summary.copy()
    out = summary.copy()
    key_cols = [
        col for col in ["checkpoint", "config_name", "exp_name", "seed", "task", "strategy"]
        if col in out.columns
    ]
    metrics = [
        col for col in ["accuracy", "macro_f1", "balanced_accuracy", "macro_sensitivity", "macro_specificity"]
        if col in out.columns
    ]
    baseline = out[out["scenario"].astype(str).eq("none")][key_cols + metrics].copy()
    baseline = baseline.rename(columns={metric: f"complete_{metric}" for metric in metrics})
    out = out.merge(baseline, on=key_cols, how="left") if key_cols and not baseline.empty else out
    for metric in metrics:
        complete_col = f"complete_{metric}"
        if complete_col in out.columns:
            out[f"{metric}_drop_from_complete"] = pd.to_numeric(out[complete_col], errors="coerce") - pd.to_numeric(out[metric], errors="coerce")
            denom = pd.to_numeric(out[complete_col], errors="coerce").replace(0, np.nan)
            out[f"{metric}_relative_drop_from_complete"] = out[f"{metric}_drop_from_complete"] / denom
    return out


def build_evidence_compensation_summary(degradation: pd.DataFrame) -> pd.DataFrame:
    if degradation.empty:
        return pd.DataFrame()
    group_cols = [
        col for col in [
            "task",
            "scenario",
            "strategy",
            *SCENARIO_METADATA_COLUMNS,
        ]
        if col in degradation.columns
    ]
    agg_cols = [
        col for col in [
            "macro_f1",
            "macro_f1_drop_from_complete",
            "accuracy_drop_from_complete",
            "macro_sensitivity_drop_from_complete",
            "macro_specificity_drop_from_complete",
            "rule_coverage",
            "hybrid_rule_rate",
            "hybrid_model_rate",
        ]
        if col in degradation.columns
    ]
    if not group_cols or not agg_cols:
        return pd.DataFrame()
    grouped = degradation.groupby(group_cols, dropna=False)
    rows = []
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row["n_profiles"] = int(len(sub))
        for col in agg_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col}__std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values([col for col in ["task", "scenario", "strategy"] if col in group_cols]).reset_index(drop=True)


def build_missingness_hybrid_reason_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    policy_specs = [
        ("hybrid_rule_first", "hybrid_decision_reason"),
        ("hybrid_rule_first_confidence_gate", "hybrid_confidence_gate_decision_reason"),
    ]
    for strategy, reason_col in policy_specs:
        if reason_col not in predictions.columns:
            continue
        sub = predictions[predictions["strategy"].astype(str).eq(strategy)].copy()
        if sub.empty:
            continue
        for col in SCENARIO_METADATA_COLUMNS:
            sub[col] = [
                scenario_metadata(str(row.get("task")), str(row.get("scenario"))).get(col, "unknown")
                for _, row in sub.iterrows()
            ]
        sub["decision_reason"] = sub[reason_col].fillna("unknown").astype(str)
        sub["model_correct"] = sub["model_pred_label"].astype(str).eq(sub["true_label"].astype(str))
        sub["warning_flag"] = sub.get("warning_reasons", pd.Series("", index=sub.index)).fillna("").astype(str).str.strip().ne("")
        group_cols = [
            col for col in [
                "task", "scenario", "strategy", *SCENARIO_METADATA_COLUMNS, "decision_reason",
            ] if col in sub.columns
        ]
        grouped = sub.groupby(group_cols, dropna=False)
        partial = grouped.agg(
            n=("decision_reason", "size"),
            accuracy=("correct", "mean"),
            model_correct_rate=("model_correct", "mean"),
            rule_coverage_rate=("baseline_covered", "mean"),
            hybrid_rule_rate=("hybrid_used_rule", "mean"),
            hybrid_model_rate=("hybrid_used_model", "mean"),
            low_confidence_abstain_rate=("hybrid_low_confidence_abstain", "mean"),
            warning_rate=("warning_flag", "mean"),
            rule_model_conflict_rate=("rule_model_conflict", "mean"),
            mean_rule_confidence=("rule_confidence", "mean"),
            mean_rule_evidence_score=("rule_evidence_score", "mean"),
            mean_model_confidence=("model_confidence", "mean"),
        ).reset_index()
        total_cols = [col for col in group_cols if col != "decision_reason"]
        partial["reason_fraction"] = partial["n"] / partial.groupby(total_cols, dropna=False)["n"].transform("sum")
        rows.append(partial)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [col for col in ["task", "scenario", "strategy", "n"] if col in out.columns]
    ascending = [True] * len(sort_cols)
    if "n" in sort_cols:
        ascending[sort_cols.index("n")] = False
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def append_rows_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    if rows is None:
        return
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    df.to_csv(
        path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        encoding="utf-8" if file_exists else "utf-8-sig",
    )


def reset_output_files(out_dir: Path, paper_tables_dir: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ARTIFICIAL_MISSINGNESS_OUTPUT_FILES:
        path = out_dir / file_name
        if path.exists():
            path.unlink()
    if paper_tables_dir is not None:
        paper_tables_dir.mkdir(parents=True, exist_ok=True)
        for file_name in PAPER_MISSINGNESS_OUTPUT_FILES:
            path = paper_tables_dir / file_name
            if path.exists():
                path.unlink()


def bool_mean_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).astype(float)


def build_missingness_hybrid_reason_partial(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    policy_specs = [
        ("hybrid_rule_first", "hybrid_decision_reason"),
        ("hybrid_rule_first_confidence_gate", "hybrid_confidence_gate_decision_reason"),
    ]
    for strategy, reason_col in policy_specs:
        if reason_col not in predictions.columns or "strategy" not in predictions.columns:
            continue
        sub = predictions[predictions["strategy"].astype(str).eq(strategy)].copy()
        if sub.empty:
            continue
        for col in SCENARIO_METADATA_COLUMNS:
            sub[col] = [
                scenario_metadata(str(row.get("task")), str(row.get("scenario"))).get(col, "unknown")
                for _, row in sub.iterrows()
            ]
        sub["decision_reason"] = sub[reason_col].fillna("unknown").astype(str)
        sub["correct_numeric"] = bool_mean_series(sub, "correct")
        sub["model_correct_numeric"] = sub["model_pred_label"].astype(str).eq(sub["true_label"].astype(str)).astype(float)
        sub["baseline_covered_numeric"] = bool_mean_series(sub, "baseline_covered")
        sub["hybrid_used_rule_numeric"] = bool_mean_series(sub, "hybrid_used_rule")
        sub["hybrid_used_model_numeric"] = bool_mean_series(sub, "hybrid_used_model")
        sub["hybrid_low_confidence_abstain_numeric"] = bool_mean_series(sub, "hybrid_low_confidence_abstain")
        sub["warning_flag_numeric"] = sub.get("warning_reasons", pd.Series("", index=sub.index)).fillna("").astype(str).str.strip().ne("").astype(float)
        sub["rule_model_conflict_numeric"] = bool_mean_series(sub, "rule_model_conflict")
        sub["rule_confidence_numeric"] = pd.to_numeric(sub.get("rule_confidence", pd.Series(np.nan, index=sub.index)), errors="coerce")
        sub["rule_evidence_score_numeric"] = pd.to_numeric(sub.get("rule_evidence_score", pd.Series(np.nan, index=sub.index)), errors="coerce")
        sub["model_confidence_numeric"] = pd.to_numeric(sub.get("model_confidence", pd.Series(np.nan, index=sub.index)), errors="coerce")
        group_cols = [
            col for col in [
                "task", "scenario", "strategy", *SCENARIO_METADATA_COLUMNS, "decision_reason",
            ] if col in sub.columns
        ]
        partial = sub.groupby(group_cols, dropna=False).agg(
            n=("decision_reason", "size"),
            correct_sum=("correct_numeric", "sum"),
            model_correct_sum=("model_correct_numeric", "sum"),
            baseline_covered_sum=("baseline_covered_numeric", "sum"),
            hybrid_used_rule_sum=("hybrid_used_rule_numeric", "sum"),
            hybrid_used_model_sum=("hybrid_used_model_numeric", "sum"),
            hybrid_low_confidence_abstain_sum=("hybrid_low_confidence_abstain_numeric", "sum"),
            warning_flag_sum=("warning_flag_numeric", "sum"),
            rule_model_conflict_sum=("rule_model_conflict_numeric", "sum"),
            rule_confidence_sum=("rule_confidence_numeric", "sum"),
            rule_confidence_count=("rule_confidence_numeric", "count"),
            rule_evidence_score_sum=("rule_evidence_score_numeric", "sum"),
            rule_evidence_score_count=("rule_evidence_score_numeric", "count"),
            model_confidence_sum=("model_confidence_numeric", "sum"),
            model_confidence_count=("model_confidence_numeric", "count"),
        ).reset_index()
        rows.append(partial)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def finalize_missingness_hybrid_reason_summary(partials: pd.DataFrame) -> pd.DataFrame:
    if partials.empty:
        return pd.DataFrame()
    group_cols = [
        col for col in [
            "task", "scenario", "strategy", *SCENARIO_METADATA_COLUMNS, "decision_reason",
        ] if col in partials.columns
    ]
    sum_cols = [col for col in partials.columns if col not in group_cols]
    grouped = partials.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    n = pd.to_numeric(grouped["n"], errors="coerce").replace(0, np.nan)
    grouped["accuracy"] = grouped["correct_sum"] / n
    grouped["model_correct_rate"] = grouped["model_correct_sum"] / n
    grouped["rule_coverage_rate"] = grouped["baseline_covered_sum"] / n
    grouped["hybrid_rule_rate"] = grouped["hybrid_used_rule_sum"] / n
    grouped["hybrid_model_rate"] = grouped["hybrid_used_model_sum"] / n
    grouped["low_confidence_abstain_rate"] = grouped["hybrid_low_confidence_abstain_sum"] / n
    grouped["warning_rate"] = grouped["warning_flag_sum"] / n
    grouped["rule_model_conflict_rate"] = grouped["rule_model_conflict_sum"] / n
    grouped["mean_rule_confidence"] = grouped["rule_confidence_sum"] / grouped["rule_confidence_count"].replace(0, np.nan)
    grouped["mean_rule_evidence_score"] = grouped["rule_evidence_score_sum"] / grouped["rule_evidence_score_count"].replace(0, np.nan)
    grouped["mean_model_confidence"] = grouped["model_confidence_sum"] / grouped["model_confidence_count"].replace(0, np.nan)
    total_cols = [col for col in group_cols if col != "decision_reason"]
    grouped["reason_fraction"] = grouped["n"] / grouped.groupby(total_cols, dropna=False)["n"].transform("sum")
    output_cols = [
        *group_cols,
        "n",
        "accuracy",
        "model_correct_rate",
        "rule_coverage_rate",
        "hybrid_rule_rate",
        "hybrid_model_rate",
        "low_confidence_abstain_rate",
        "warning_rate",
        "rule_model_conflict_rate",
        "mean_rule_confidence",
        "mean_rule_evidence_score",
        "mean_model_confidence",
        "reason_fraction",
    ]
    out = grouped[output_cols].copy()
    sort_cols = [col for col in ["task", "scenario", "strategy", "n"] if col in out.columns]
    ascending = [True] * len(sort_cols)
    if "n" in sort_cols:
        ascending[sort_cols.index("n")] = False
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def build_missingness_hybrid_reason_summary_from_csv(predictions_path: Path, chunksize: int = PREDICTION_CHUNKSIZE) -> pd.DataFrame:
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        return pd.DataFrame()
    partials = []
    try:
        for chunk in pd.read_csv(predictions_path, chunksize=chunksize, low_memory=False):
            partial = build_missingness_hybrid_reason_partial(chunk)
            if not partial.empty:
                partials.append(partial)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if not partials:
        return pd.DataFrame()
    return finalize_missingness_hybrid_reason_summary(pd.concat(partials, ignore_index=True))
def parse_checkpoint_identity(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    exp_seed = path.parent.name
    match = re.match(r"(?P<exp>.+)_seed_(?P<seed>\d+)$", exp_seed)
    exp_name = match.group("exp") if match else exp_seed
    seed = int(match.group("seed")) if match else None
    config_name = path.parent.parent.name if path.parent.parent != path.parent else None
    return {
        "checkpoint": str(path),
        "config_name": config_name,
        "exp_name": exp_name,
        "seed": seed,
    }


def scenario_requires_model_fallback(task_name: str, scenario: str) -> bool:
    return scenario in SCENARIO_FORCE_MODEL_FALLBACK.get(task_name, set())


def apply_scenario_rule_policy(rule_info: dict, task_name: str, scenario: str) -> dict:
    out = dict(rule_info)
    if not scenario_requires_model_fallback(task_name, scenario):
        out["scenario_forced_model_fallback"] = False
        return out

    warnings = [part for part in str(out.get("warning_reasons") or "").split(";") if part]
    warnings.append("scenario_forced_model_fallback")
    out["baseline_covered"] = False
    out["complete_for_rule"] = False
    out["abstain_rule_label"] = INSUFFICIENT_EVIDENCE_LABEL
    out["rule_confidence"] = min(float(out.get("rule_confidence", 0.0) or 0.0), 0.5)
    out["warning_reasons"] = ";".join(dict.fromkeys(warnings))
    out["scenario_forced_model_fallback"] = True
    return out


def strategy_predictions(
    rule_info: dict,
    model_pred: str,
    threshold: float,
    model_confidence: float | None = None,
    model_confidence_threshold: float = 0.6,
) -> dict:
    rule_confidence = float(rule_info.get("rule_confidence", 0.0) or 0.0)
    baseline_covered = bool(rule_info.get("baseline_covered", False))
    complete_for_rule = bool(rule_info.get("complete_for_rule", baseline_covered))
    forced = str(rule_info.get("forced_rule_label") or INSUFFICIENT_EVIDENCE_LABEL)
    abstain = str(rule_info.get("abstain_rule_label") or INSUFFICIENT_EVIDENCE_LABEL)
    rule_label_available = forced not in {"", "None", INSUFFICIENT_EVIDENCE_LABEL}
    warning_text = str(rule_info.get("warning_reasons") or "").strip(";")
    hard_warning = has_hard_rule_warning(warning_text)
    use_rule = (
        rule_label_available
        and baseline_covered
        and complete_for_rule
        and rule_confidence >= float(threshold)
        and not hard_warning
    )
    confidence_threshold = max(0.0, min(1.0, float(model_confidence_threshold or 0.0)))
    confidence_value = None if model_confidence is None else float(model_confidence)
    low_confidence = (
        (not use_rule)
        and confidence_value is not None
        and confidence_value < confidence_threshold
    )
    hard_warning_blocked_rule = bool(rule_label_available and hard_warning and not use_rule)
    if use_rule:
        reason = "rule_score_ge_threshold_with_warning" if warning_text else "rule_score_ge_threshold"
    elif bool(rule_info.get("scenario_forced_model_fallback", False)):
        reason = "model_fallback_scenario_forced_missingness"
    elif not rule_label_available:
        reason = "model_fallback_no_rule_prediction"
    elif not baseline_covered:
        reason = "model_fallback_rule_not_covered"
    elif not complete_for_rule:
        reason = "model_fallback_incomplete_rule_data"
    elif hard_warning_blocked_rule:
        reason = "model_fallback_hard_rule_warning"
    elif rule_confidence < float(threshold):
        reason = "model_fallback_low_rule_score"
    else:
        reason = "model_fallback"
    confidence_gate_reason = "abstain_low_model_confidence" if low_confidence else reason
    return {
        "model_only": str(model_pred),
        "rule_forced": forced,
        "rule_abstain_as_error": abstain,
        "hybrid_rule_first": forced if use_rule else str(model_pred),
        "hybrid_rule_first_confidence_gate": forced if use_rule else (INSUFFICIENT_EVIDENCE_LABEL if low_confidence else str(model_pred)),
        "hybrid_used_rule": bool(use_rule),
        "hybrid_used_model": not bool(use_rule),
        "hybrid_confidence_gate_used_rule": bool(use_rule),
        "hybrid_confidence_gate_used_model": (not bool(use_rule)) and (not bool(low_confidence)),
        "hybrid_low_confidence_abstain": bool(low_confidence),
        "hybrid_hard_warning_blocked_rule": bool(hard_warning_blocked_rule),
        "hybrid_decision_reason": reason,
        "hybrid_confidence_gate_decision_reason": confidence_gate_reason,
    }


def scenario_summary_rows(
    base_info: dict,
    task_name: str,
    scenario: str,
    y_true: list,
    strategy_to_pred: dict[str, list],
    hybrid_rule_flags: list[bool],
    rule_covered_flags: list[bool],
    hybrid_gate_rule_flags: list[bool] | None = None,
    hybrid_gate_model_flags: list[bool] | None = None,
    hybrid_low_confidence_flags: list[bool] | None = None,
) -> list[dict]:
    rows = []
    for strategy, y_pred in strategy_to_pred.items():
        row = {**base_info}
        row["scenario"] = scenario
        row["strategy"] = strategy
        row.update(scenario_metadata(task_name, scenario))
        row.update(metric_dict(task_name, y_true, y_pred))
        row["rule_coverage"] = float(np.mean(rule_covered_flags)) if rule_covered_flags else None
        row["hybrid_rule_rate"] = None
        row["hybrid_model_rate"] = None
        row["low_confidence_abstain_rate"] = None
        if strategy == "hybrid_rule_first" and hybrid_rule_flags:
            row["hybrid_rule_rate"] = float(np.mean(hybrid_rule_flags))
            row["hybrid_model_rate"] = float(1.0 - np.mean(hybrid_rule_flags))
            row["low_confidence_abstain_rate"] = 0.0
        elif strategy == "hybrid_rule_first_confidence_gate":
            if hybrid_gate_rule_flags:
                row["hybrid_rule_rate"] = float(np.mean(hybrid_gate_rule_flags))
            if hybrid_gate_model_flags:
                row["hybrid_model_rate"] = float(np.mean(hybrid_gate_model_flags))
            if hybrid_low_confidence_flags:
                row["low_confidence_abstain_rate"] = float(np.mean(hybrid_low_confidence_flags))
        rows.append(row)
    return rows


def load_task_data(data_dir: Path, task_name: str):
    info = TASK_INFO[task_name]
    raw = load_tabular_data(data_dir / info["csv"])
    df = prepare_task_dataframe(task_name, raw)
    df, _ = clean_label_columns(df, info["label_cols"])
    return df


def resolve_checkpoints(args) -> list[str]:
    paths = resolve_checkpoint_paths(
        checkpoint=args.checkpoint,
        checkpoints=args.checkpoints,
        checkpoint_glob=args.checkpoint_glob,
        default_checkpoint=DEFAULT_CHECKPOINT,
        search_defaults=True,
    )
    return [str(path) for path in paths]


def run_experiment(args):
    device = resolve_torch_device(args.device)
    print(f"[artificial_missingness_v74] Requested device={args.device}; resolved device={device}")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    paper_tables_dir = Path(args.paper_tables_dir) if getattr(args, "paper_tables_dir", "") else None
    reset_output_files(out_dir, paper_tables_dir)

    summary_path = out_dir / "artificial_missingness_summary.csv"
    degradation_path = out_dir / "artificial_missingness_degradation_summary.csv"
    per_class_path = out_dir / "artificial_missingness_per_class.csv"
    compensation_path = out_dir / "evidence_compensation_summary.csv"
    reason_path = out_dir / "missingness_hybrid_reason_summary.csv"
    taxonomy_path = out_dir / "clinical_missingness_taxonomy.csv"
    predictions_path = out_dir / "artificial_missingness_predictions.csv"

    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    checkpoints = resolve_checkpoints(args)

    for checkpoint in checkpoints:
        model, meta = load_model(checkpoint, device)
        base_info = parse_checkpoint_identity(checkpoint)
        if meta.get("exp_name"):
            base_info["exp_name"] = meta.get("exp_name")
        if meta.get("seed") is not None:
            base_info["seed"] = meta.get("seed")
        if meta.get("run_config", {}).get("name"):
            base_info["config_name"] = meta["run_config"]["name"]

        for task_name in requested_tasks:
            base_df = load_task_data(data_dir, task_name)
            label_col = TASK_INFO[task_name]["label_cols"][0]
            for scenario in task_scenarios(task_name):
                scenario_df = apply_artificial_missingness(base_df, task_name, scenario)
                y_true = scenario_df[label_col].astype(str).tolist()
                y_model, confidences = predict_task(
                    model,
                    meta,
                    scenario_df,
                    task_name,
                    args.batch_size,
                    device,
                )
                strategy_to_pred = {
                    "model_only": [],
                    "rule_forced": [],
                    "rule_abstain_as_error": [],
                    "hybrid_rule_first": [],
                    "hybrid_rule_first_confidence_gate": [],
                }
                hybrid_rule_flags = []
                hybrid_gate_rule_flags = []
                hybrid_gate_model_flags = []
                hybrid_low_confidence_flags = []
                rule_covered_flags = []
                scenario_pred_rows = []

                for idx, (truth, model_pred, conf) in enumerate(zip(y_true, y_model, confidences)):
                    row = scenario_df.iloc[idx]
                    rule_info = clinical_warning_summary(task_name, row.to_dict(), model_pred=model_pred)
                    rule_info = apply_scenario_rule_policy(rule_info, task_name, scenario)
                    preds = strategy_predictions(
                        rule_info,
                        model_pred,
                        args.rule_confidence_threshold,
                        model_confidence=conf,
                        model_confidence_threshold=args.model_confidence_threshold,
                    )
                    rule_covered_flags.append(bool(rule_info.get("baseline_covered", False)))
                    hybrid_rule_flags.append(bool(preds["hybrid_used_rule"]))
                    hybrid_gate_rule_flags.append(bool(preds["hybrid_confidence_gate_used_rule"]))
                    hybrid_gate_model_flags.append(bool(preds["hybrid_confidence_gate_used_model"]))
                    hybrid_low_confidence_flags.append(bool(preds["hybrid_low_confidence_abstain"]))
                    for strategy in strategy_to_pred:
                        strategy_to_pred[strategy].append(preds[strategy])
                        scenario_pred_rows.append({
                            **base_info,
                            "scenario": scenario,
                            "strategy": strategy,
                            "task": task_name,
                            "label_col": label_col,
                            "row_index": idx,
                            "case_id": row.get("case_id"),
                            "ear_side": row.get("ear_side"),
                            "true_label": truth,
                            "pred_label": preds[strategy],
                            "correct": bool(truth == preds[strategy]),
                            "model_pred_label": model_pred,
                            "model_confidence": float(conf),
                            "forced_rule_label": rule_info.get("forced_rule_label"),
                            "abstain_rule_label": rule_info.get("abstain_rule_label"),
                            "baseline_covered": bool(rule_info.get("baseline_covered", False)),
                            "complete_for_rule": bool(rule_info.get("complete_for_rule", rule_info.get("baseline_covered", False))),
                            "rule_confidence": float(rule_info.get("rule_confidence", 0.0) or 0.0),
                            "evidence_status": rule_info.get("evidence_status"),
                            "compatible_labels": rule_info.get("compatible_labels"),
                            "rule_model_conflict": bool(rule_info.get("rule_model_conflict", False)),
                            "warning_reasons": rule_info.get("warning_reasons"),
                            "rule_evidence_score": float(rule_info.get("rule_evidence_score", rule_info.get("rule_confidence", 0.0)) or 0.0),
                            "score_deductions": rule_info.get("score_deductions"),
                            "scenario_forced_model_fallback": bool(rule_info.get("scenario_forced_model_fallback", False)),
                            "hybrid_used_rule": bool(preds["hybrid_used_rule"]),
                            "hybrid_used_model": bool(preds["hybrid_used_model"]),
                            "hybrid_confidence_gate_used_rule": bool(preds["hybrid_confidence_gate_used_rule"]),
                            "hybrid_confidence_gate_used_model": bool(preds["hybrid_confidence_gate_used_model"]),
                            "hybrid_low_confidence_abstain": bool(preds["hybrid_low_confidence_abstain"]),
                            "hybrid_hard_warning_blocked_rule": bool(preds.get("hybrid_hard_warning_blocked_rule", False)),
                            "hybrid_decision_reason": preds["hybrid_decision_reason"],
                            "hybrid_confidence_gate_decision_reason": preds["hybrid_confidence_gate_decision_reason"],
                            "model_confidence_threshold": float(args.model_confidence_threshold),
                        })

                scenario_rows = scenario_summary_rows(
                    base_info,
                    task_name,
                    scenario,
                    y_true,
                    strategy_to_pred,
                    hybrid_rule_flags,
                    rule_covered_flags,
                    hybrid_gate_rule_flags,
                    hybrid_gate_model_flags,
                    hybrid_low_confidence_flags,
                )
                scenario_per_class_rows = []
                for strategy, y_pred in strategy_to_pred.items():
                    scenario_per_class_rows.extend(
                        class_metric_rows(base_info, task_name, scenario, strategy, y_true, y_pred)
                    )

                append_rows_csv(summary_path, scenario_rows)
                append_rows_csv(per_class_path, scenario_per_class_rows)
                append_rows_csv(predictions_path, scenario_pred_rows)

                del scenario_df
                del y_true, y_model, confidences, strategy_to_pred
                del hybrid_rule_flags, hybrid_gate_rule_flags, hybrid_gate_model_flags, hybrid_low_confidence_flags, rule_covered_flags
                del scenario_rows, scenario_per_class_rows, scenario_pred_rows
                gc.collect()
            del base_df

        del model, meta
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    degradation = add_complete_baseline_deltas(summary)
    compensation = build_evidence_compensation_summary(degradation)
    missingness_reason = build_missingness_hybrid_reason_summary_from_csv(predictions_path)
    taxonomy = build_clinical_missingness_taxonomy(requested_tasks)

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    degradation.to_csv(degradation_path, index=False, encoding="utf-8-sig")
    compensation.to_csv(compensation_path, index=False, encoding="utf-8-sig")
    missingness_reason.to_csv(reason_path, index=False, encoding="utf-8-sig")
    taxonomy.to_csv(taxonomy_path, index=False, encoding="utf-8-sig")

    if paper_tables_dir is not None:
        paper_tables_dir.mkdir(parents=True, exist_ok=True)
        degradation.to_csv(paper_tables_dir / "missingness_degradation_summary.csv", index=False, encoding="utf-8-sig")
        compensation.to_csv(paper_tables_dir / "missingness_evidence_compensation_summary.csv", index=False, encoding="utf-8-sig")
        missingness_reason.to_csv(paper_tables_dir / "missingness_hybrid_reason_summary.csv", index=False, encoding="utf-8-sig")
        taxonomy.to_csv(paper_tables_dir / "clinical_missingness_taxonomy.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "checkpoints": checkpoints,
        "tasks": requested_tasks,
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "resolved_device": str(device),
        "rule_confidence_threshold": args.rule_confidence_threshold,
        "model_confidence_threshold": args.model_confidence_threshold,
        "scenarios": {task: task_scenarios(task) for task in requested_tasks},
        "strategies": [
            "model_only",
            "rule_forced",
            "rule_abstain_as_error",
            "hybrid_rule_first",
            "hybrid_rule_first_confidence_gate",
        ],
        "scenario_metadata": SCENARIO_METADATA,
        "scenario_force_model_fallback": {task: sorted(values) for task, values in SCENARIO_FORCE_MODEL_FALLBACK.items()},
        "write_mode": "streaming_per_scenario",
        "prediction_chunksize": PREDICTION_CHUNKSIZE,
        "outputs": {
            "summary": str(summary_path),
            "degradation_summary": str(degradation_path),
            "per_class": str(per_class_path),
            "evidence_compensation": str(compensation_path),
            "hybrid_reason_summary": str(reason_path),
            "clinical_missingness_taxonomy": str(taxonomy_path),
            "predictions": str(predictions_path),
        },
    }
    (out_dir / "artificial_missingness_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate support-free robustness under artificial missingness stress scenarios.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint paths.")
    parser.add_argument("--checkpoint-glob", default=None, help="Glob such as results_v74/five_runs/**/best_model.pth.")
    parser.add_argument("--output-dir", default="results_v74/artificial_missingness")
    parser.add_argument("--paper-tables-dir", default="", help="Optional paper_v74/tables directory for paper-ready missingness summaries.")
    parser.add_argument("--tasks", default="Task1,Task2,Task3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rule-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--model-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--device", default="auto", help="Device for neural inference: auto, cpu, cuda, or cuda:<index>.")
    args = parser.parse_args()
    summary = run_experiment(args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
