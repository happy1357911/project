import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PREDICTION_FILENAMES = {
    "configured": "prediction_rows.csv",
    "no_support": "prediction_rows_no_support.csv",
    "locked_test": "prediction_rows_locked_test_no_support.csv",
}
INSUFFICIENT_EVIDENCE_LABEL = "INSUFFICIENT_EVIDENCE"


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def safe_numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = list(cols)
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def numeric_column(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def count_flag_columns(df: pd.DataFrame, suffixes: Iterable[str]) -> pd.Series:
    cols = [
        col for col in df.columns
        if col.startswith("feature__") and any(col.endswith(suffix) for suffix in suffixes)
    ]
    if not cols:
        return pd.Series(0, index=df.index, dtype=int)
    return safe_numeric_frame(df, cols).ge(0.5).sum(axis=1).astype(int)


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["correct"] = parse_bool_series(out["correct"]) if "correct" in out.columns else (out["true_label"] == out["pred_label"])
    out["confidence"] = numeric_column(out, "confidence")
    out["expert_confidence"] = numeric_column(out, "expert_confidence")
    out["confidence_bucket"] = pd.cut(
        out["confidence"],
        bins=[-np.inf, 0.5, 0.7, 0.9, np.inf],
        labels=["<=0.50", "0.50-0.70", "0.70-0.90", ">0.90"],
    ).astype(str)
    out["expert_confidence_bucket"] = pd.cut(
        out["expert_confidence"],
        bins=[-np.inf, 0.3, 0.6, 0.9, np.inf],
        labels=["<=0.30", "0.30-0.60", "0.60-0.90", ">0.90"],
    ).astype(str)

    pta = numeric_column(out, "feature__ac_PTA")
    out["task1_pta_bucket"] = pd.cut(
        pta,
        bins=[-np.inf, 20, 35, 50, 65, 80, 95, np.inf],
        labels=["<20", "20-34", "35-49", "50-64", "65-79", "80-94", ">=95"],
    ).astype(str)

    out["task2_bc_missing_count"] = count_flag_columns(out, ["Hz_missing"])
    out["task2_abg_missing_count"] = count_flag_columns(out, ["Hz_missing"])
    out["task2_ac_nr_count"] = count_flag_columns(out, ["Hz_nr"])
    out["task2_bc_nr_count"] = count_flag_columns(out, ["Hz_nr"])
    out["task2_abg_censored_count"] = count_flag_columns(out, ["Hz_censored"])
    task2_feature_cols = [col for col in out.columns if col.startswith("feature__")]
    out["task2_bc_missing_count"] = safe_numeric_frame(
        out,
        [col for col in task2_feature_cols if col.startswith("feature__bc_") and col.endswith("Hz_missing")],
    ).ge(0.5).sum(axis=1).astype(int)
    out["task2_abg_missing_count"] = safe_numeric_frame(
        out,
        [col for col in task2_feature_cols if col.startswith("feature__abg_") and col.endswith("Hz_missing")],
    ).ge(0.5).sum(axis=1).astype(int)
    out["task2_ac_nr_count"] = safe_numeric_frame(
        out,
        [col for col in task2_feature_cols if col.startswith("feature__ac_") and col.endswith("Hz_nr")],
    ).ge(0.5).sum(axis=1).astype(int)
    out["task2_bc_nr_count"] = safe_numeric_frame(
        out,
        [col for col in task2_feature_cols if col.startswith("feature__bc_") and col.endswith("Hz_nr")],
    ).ge(0.5).sum(axis=1).astype(int)
    out["task2_abg_censored_count"] = safe_numeric_frame(
        out,
        [col for col in task2_feature_cols if col.startswith("feature__abg_") and col.endswith("Hz_censored")],
    ).ge(0.5).sum(axis=1).astype(int)
    borderline_counts = pd.Series(0, index=out.index, dtype=int)
    for hz in [500, 1000, 2000, 4000]:
        abg_col = f"feature__abg_{hz}Hz"
        missing_col = f"feature__abg_{hz}Hz_missing"
        censored_col = f"feature__abg_{hz}Hz_censored"
        if abg_col not in out.columns:
            continue
        abg = numeric_column(out, abg_col)
        missing = numeric_column(out, missing_col, default=0.0).ge(0.5)
        censored = numeric_column(out, censored_col, default=0.0).ge(0.5)
        borderline_counts += (abg.between(8.0, 12.0, inclusive="both") & ~missing & ~censored).astype(int)
    out["task2_abg_borderline_count"] = borderline_counts.astype(int)
    out["task2_has_bc_missing"] = out["task2_bc_missing_count"].gt(0)
    out["task2_has_ac_nr"] = out["task2_ac_nr_count"].gt(0)
    out["task2_has_bc_nr"] = out["task2_bc_nr_count"].gt(0)
    out["task2_has_any_nr"] = out[["task2_ac_nr_count", "task2_bc_nr_count"]].sum(axis=1).gt(0)
    out["task2_has_abg_censored"] = out["task2_abg_censored_count"].gt(0)
    out["task2_has_abg_borderline"] = out["task2_abg_borderline_count"].gt(0)
    out["task2_has_missing_or_censored"] = (
        out[["task2_bc_missing_count", "task2_abg_missing_count", "task2_abg_censored_count"]]
        .sum(axis=1)
        .gt(0)
    )

    out["task3_np_count"] = count_flag_columns(out, ["_np_zero"])
    out["task3_missing_count"] = count_flag_columns(out, ["_missing_zero"])
    out["task3_has_np"] = out["task3_np_count"].gt(0)
    out["task3_has_missing"] = out["task3_missing_count"].gt(0)
    return out


def discover_prediction_files(results_dir: Path, mode: str) -> List[Path]:
    if mode == "both":
        filenames = [PREDICTION_FILENAMES["configured"], PREDICTION_FILENAMES["no_support"]]
    elif mode == "all":
        filenames = PREDICTION_FILENAMES.values()
    elif mode in PREDICTION_FILENAMES:
        filenames = [PREDICTION_FILENAMES[mode]]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    paths = []
    for filename in filenames:
        paths.extend(results_dir.rglob(filename))
    return sorted(set(paths))


def parse_run_identity(path: Path, results_dir: Path) -> Dict[str, object]:
    run_dir = path.parent
    run_name = run_dir.name
    exp_name = run_name
    seed = None
    if "_seed_" in run_name:
        exp_name, seed_part = run_name.rsplit("_seed_", 1)
        try:
            seed = int(seed_part)
        except ValueError:
            seed = None
    config_name = run_dir.parent.name if run_dir.parent != results_dir else ""
    return {
        "config_name": config_name,
        "run_dir": str(run_dir),
        "exp_name": exp_name,
        "seed": seed,
        "prediction_file": path.name,
    }


def load_prediction_rows(results_dir: Path, mode: str) -> pd.DataFrame:
    frames = []
    for path in discover_prediction_files(results_dir, mode):
        df = pd.read_csv(path)
        identity = parse_run_identity(path, results_dir)
        for key, value in identity.items():
            df.insert(0, key, value)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return add_subgroup_columns(pd.concat(frames, ignore_index=True))


def load_rule_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep_cols = [
        "task",
        "case_id",
        "ear_side",
        "true_label",
        "forced_pred_label",
        "abstain_pred_label",
        "baseline_covered",
        "complete_for_rule",
        "evidence_status",
        "rule_confidence",
        "warning_reasons",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep_cols].copy()


def attach_rule_predictions(model_df: pd.DataFrame, rule_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty or rule_df.empty:
        return model_df
    left = model_df.copy()
    right = rule_df.copy()
    for col in ["task", "case_id", "ear_side", "true_label"]:
        if col in left.columns:
            left[col] = left[col].astype(str)
        if col in right.columns:
            right[col] = right[col].astype(str)
    rule_cols = [
        "forced_pred_label",
        "abstain_pred_label",
        "baseline_covered",
        "complete_for_rule",
        "evidence_status",
        "rule_confidence",
        "warning_reasons",
    ]
    left = left.drop(columns=[col for col in rule_cols if col in left.columns], errors="ignore")
    merged = left.merge(
        right,
        on=["task", "case_id", "ear_side", "true_label"],
        how="left",
        suffixes=("", "_rule"),
    )
    if "baseline_covered" in merged.columns:
        merged["baseline_covered"] = parse_bool_series(merged["baseline_covered"])
    if "complete_for_rule" in merged.columns:
        complete_raw = merged["complete_for_rule"]
        complete = parse_bool_series(complete_raw)
        if "baseline_covered" in merged.columns:
            complete = complete.mask(complete_raw.isna(), merged["baseline_covered"])
        merged["complete_for_rule"] = complete.astype(bool)
    if "rule_confidence" in merged.columns:
        merged["rule_confidence"] = pd.to_numeric(merged["rule_confidence"], errors="coerce")
    return merged


def summarize_group(df: pd.DataFrame, group_type: str, group_cols: List[str]) -> List[Dict[str, object]]:
    if df.empty:
        return []
    rows = []
    base_cols = ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "task", "label_col"]
    cols = base_cols + group_cols
    existing_cols = [col for col in cols if col in df.columns]
    for keys, sub in df.groupby(existing_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(existing_cols, keys)}
        labels = sorted(set(sub["true_label"].astype(str)) | set(sub["pred_label"].astype(str)))
        row.update({
            "group_type": group_type,
            "n": int(len(sub)),
            "errors": int((~sub["correct"]).sum()),
            "accuracy": float(sub["correct"].mean()) if len(sub) else None,
            "macro_f1": float(
                f1_score(
                    sub["true_label"].astype(str),
                    sub["pred_label"].astype(str),
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ) if labels else None,
            "mean_confidence": float(sub["confidence"].mean()) if sub["confidence"].notna().any() else None,
            "mean_expert_confidence": (
                float(sub["expert_confidence"].mean())
                if "expert_confidence" in sub.columns and sub["expert_confidence"].notna().any()
                else None
            ),
        })
        rows.append(row)
    return rows


def build_subgroup_metrics(df: pd.DataFrame) -> pd.DataFrame:
    subgroup_specs = [
        ("overall", []),
        ("true_label", ["true_label"]),
        ("pred_label", ["pred_label"]),
        ("ear_side", ["ear_side"]),
        ("confidence_bucket", ["confidence_bucket"]),
        ("expert_confidence_bucket", ["expert_confidence_bucket"]),
        ("task1_pta_bucket", ["task1_pta_bucket"]),
        ("task2_has_bc_missing", ["task2_has_bc_missing"]),
        ("task2_has_ac_nr", ["task2_has_ac_nr"]),
        ("task2_has_bc_nr", ["task2_has_bc_nr"]),
        ("task2_has_any_nr", ["task2_has_any_nr"]),
        ("task2_has_abg_censored", ["task2_has_abg_censored"]),
        ("task2_has_abg_borderline", ["task2_has_abg_borderline"]),
        ("task2_has_missing_or_censored", ["task2_has_missing_or_censored"]),
        ("task3_has_np", ["task3_has_np"]),
        ("task3_has_missing", ["task3_has_missing"]),
    ]
    rows = []
    for group_type, cols in subgroup_specs:
        task_df = df.copy()
        if group_type.startswith("task1_"):
            task_df = task_df[task_df["task"] == "Task1"]
        elif group_type.startswith("task2_"):
            task_df = task_df[task_df["task"] == "Task2"]
        elif group_type.startswith("task3_"):
            task_df = task_df[task_df["task"] == "Task3"]
        rows.extend(summarize_group(task_df, group_type, cols))
    return pd.DataFrame(rows)


def build_confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "task", "label_col", "true_label", "pred_label"]
    grouped = (
        df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["task", "label_col", "n"], ascending=[True, True, False])
    )
    grouped["is_error_pair"] = grouped["true_label"].astype(str) != grouped["pred_label"].astype(str)
    return grouped


def add_rule_conflict_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expert_target_numeric"] = numeric_column(out, "expert_target")
    out["expert_confidence_numeric"] = numeric_column(out, "expert_confidence")
    out["label_model_match"] = out["true_label"].astype(str).eq(out["pred_label"].astype(str))
    out["rule_supports_model"] = out["expert_target_numeric"].ge(0.5)
    out["rule_rejects_model"] = out["expert_target_numeric"].le(0.0)
    out["rule_uncertain"] = (
        out["expert_target_numeric"].between(0.0, 0.5, inclusive="neither")
        | out["expert_confidence_numeric"].lt(0.6)
        | out["expert_target_numeric"].isna()
    )
    conditions = [
        out["label_model_match"] & out["rule_supports_model"],
        out["label_model_match"] & out["rule_rejects_model"],
        (~out["label_model_match"]) & out["rule_supports_model"],
        (~out["label_model_match"]) & out["rule_rejects_model"],
        out["rule_uncertain"],
    ]
    choices = [
        "model_correct_rule_supports_model",
        "model_correct_rule_rejects_model",
        "model_wrong_rule_supports_model",
        "model_wrong_rule_rejects_model",
        "rule_uncertain_or_low_confidence",
    ]
    out["rule_model_conflict_type"] = np.select(conditions, choices, default="unclassified")
    return out


def build_rule_model_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "expert_target" not in df.columns:
        return pd.DataFrame()
    out = add_rule_conflict_columns(df)
    conflict_types = {
        "model_correct_rule_rejects_model",
        "model_wrong_rule_supports_model",
        "rule_uncertain_or_low_confidence",
    }
    keep = out["rule_model_conflict_type"].isin(conflict_types)
    detail_cols = [
        "config_name",
        "exp_name",
        "seed",
        "prediction_file",
        "evaluation_mode",
        "task",
        "label_col",
        "case_id",
        "ear_side",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "expert_target",
        "expert_confidence",
        "reward_pred",
        "rule_model_conflict_type",
    ]
    existing_cols = [col for col in detail_cols if col in out.columns]
    return out.loc[keep, existing_cols].sort_values(
        ["task", "rule_model_conflict_type", "confidence"],
        ascending=[True, True, False],
    )


def build_rule_model_conflict_summary(conflict_df: pd.DataFrame) -> pd.DataFrame:
    if conflict_df.empty:
        return pd.DataFrame()
    cols = [
        "config_name",
        "exp_name",
        "seed",
        "prediction_file",
        "evaluation_mode",
        "task",
        "label_col",
        "rule_model_conflict_type",
    ]
    existing_cols = [col for col in cols if col in conflict_df.columns]
    grouped = (
        conflict_df.groupby(existing_cols, dropna=False)
        .agg(
            n=("rule_model_conflict_type", "size"),
            mean_confidence=("confidence", "mean"),
            mean_expert_confidence=("expert_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["task", "n"], ascending=[True, False])
    )
    return grouped


def add_rule_true_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "forced_pred_label" not in out.columns:
        return pd.DataFrame()
    out["rule_label"] = out["forced_pred_label"].fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
    out["true_label"] = out["true_label"].astype(str)
    out["model_pred"] = out["pred_label"].astype(str)
    out["rule_available"] = (
        out["rule_label"].ne(INSUFFICIENT_EVIDENCE_LABEL)
        & out["rule_label"].str.strip().ne("")
        & out["rule_label"].str.lower().ne("nan")
    )
    out["model_correct"] = out["true_label"].eq(out["model_pred"])
    out["rule_correct"] = out["rule_available"] & out["true_label"].eq(out["rule_label"])
    out["rule_model_agree"] = out["rule_available"] & out["model_pred"].eq(out["rule_label"])
    out["rule_model_disagree"] = out["rule_available"] & (~out["rule_model_agree"])
    out["rule_abstained"] = (
        (~out["rule_available"])
        | out.get("abstain_pred_label", pd.Series("", index=out.index)).astype(str).eq(INSUFFICIENT_EVIDENCE_LABEL)
    )
    out["model_correct_when_rule_abstained"] = out["rule_abstained"] & out["model_correct"]
    out["model_wrong_when_rule_covered"] = out["rule_available"] & (~out["model_correct"])
    conditions = [
        ~out["rule_available"],
        out["model_correct"] & out["rule_correct"] & out["rule_model_agree"],
        (~out["model_correct"]) & out["rule_correct"],
        out["model_correct"] & (~out["rule_correct"]),
        (~out["model_correct"]) & (~out["rule_correct"]) & out["rule_model_agree"],
        (~out["model_correct"]) & (~out["rule_correct"]) & (~out["rule_model_agree"]),
    ]
    choices = [
        "rule_unavailable",
        "all_agree_correct",
        "model_wrong_rule_correct",
        "model_correct_rule_wrong",
        "model_rule_agree_wrong",
        "model_rule_disagree_both_wrong",
    ]
    out["rule_true_model_conflict_type"] = np.select(conditions, choices, default="unclassified")
    return out


def build_rule_true_model_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    out = add_rule_true_model_columns(df)
    if out.empty:
        return pd.DataFrame()
    benign = {"all_agree_correct", "rule_unavailable"}
    keep = ~out["rule_true_model_conflict_type"].isin(benign)
    detail_cols = [
        "config_name",
        "exp_name",
        "seed",
        "prediction_file",
        "evaluation_mode",
        "task",
        "label_col",
        "case_id",
        "ear_side",
        "true_label",
        "model_pred",
        "rule_label",
        "abstain_pred_label",
        "model_correct",
        "rule_correct",
        "rule_model_agree",
        "rule_model_disagree",
        "rule_abstained",
        "model_correct_when_rule_abstained",
        "model_wrong_when_rule_covered",
        "baseline_covered",
        "complete_for_rule",
        "evidence_status",
        "rule_confidence",
        "warning_reasons",
        "confidence",
        "rule_true_model_conflict_type",
    ]
    existing_cols = [col for col in detail_cols if col in out.columns]
    return out.loc[keep, existing_cols].sort_values(
        ["task", "rule_true_model_conflict_type", "confidence"],
        ascending=[True, True, False],
    )


def build_rule_true_model_conflict_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = add_rule_true_model_columns(df)
    if out.empty:
        return pd.DataFrame()
    group_cols = [
        "config_name",
        "exp_name",
        "seed",
        "prediction_file",
        "evaluation_mode",
        "task",
        "label_col",
        "rule_true_model_conflict_type",
    ]
    existing_cols = [col for col in group_cols if col in out.columns]
    grouped = (
        out.groupby(existing_cols, dropna=False)
        .agg(
            n=("rule_true_model_conflict_type", "size"),
            model_correct_rate=("model_correct", "mean"),
            rule_correct_rate=("rule_correct", "mean"),
            rule_model_agree_rate=("rule_model_agree", "mean"),
            rule_model_disagree_rate=("rule_model_disagree", "mean"),
            rule_abstained_rate=("rule_abstained", "mean"),
            model_correct_when_rule_abstained_rate=("model_correct_when_rule_abstained", "mean"),
            model_wrong_when_rule_covered_rate=("model_wrong_when_rule_covered", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_rule_confidence=("rule_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["task", "n"], ascending=[True, False])
    )
    return grouped


def add_three_way_rule_columns(df: pd.DataFrame) -> pd.DataFrame:
    return add_rule_true_model_columns(df)


def build_three_way_conflict_summary(df: pd.DataFrame) -> pd.DataFrame:
    return build_rule_true_model_conflict_summary(df)


def build_task2_clinical_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "task" not in df.columns:
        return pd.DataFrame()
    task2 = df[df["task"] == "Task2"].copy()
    if task2.empty:
        return pd.DataFrame()
    subgroup_cols = [
        "task2_has_bc_missing",
        "task2_has_ac_nr",
        "task2_has_bc_nr",
        "task2_has_abg_censored",
        "task2_has_abg_borderline",
        "task2_has_missing_or_censored",
    ]
    rows = []
    base_cols = ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "label_col"]
    for subgroup_col in subgroup_cols:
        existing = [col for col in base_cols + [subgroup_col] if col in task2.columns]
        for keys, sub in task2.groupby(existing, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: value for col, value in zip(existing, keys)}
            labels = sorted(set(sub["true_label"].astype(str)) | set(sub["pred_label"].astype(str)))
            row.update({
                "subgroup": subgroup_col,
                "n": int(len(sub)),
                "errors": int((~sub["correct"]).sum()),
                "accuracy": float(sub["correct"].mean()) if len(sub) else None,
                "macro_f1": float(
                    f1_score(
                        sub["true_label"].astype(str),
                        sub["pred_label"].astype(str),
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ) if labels else None,
                "mean_confidence": float(sub["confidence"].mean()) if sub["confidence"].notna().any() else None,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def build_task2_confusion_focus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "task" not in df.columns:
        return pd.DataFrame()
    task2 = df[df["task"] == "Task2"].copy()
    if task2.empty:
        return pd.DataFrame()
    task2["is_chl_mhl_snhl_confusion"] = (
        task2["true_label"].astype(str).isin(["CHL", "MHL", "SNHL"])
        & task2["pred_label"].astype(str).isin(["CHL", "MHL", "SNHL"])
        & task2["true_label"].astype(str).ne(task2["pred_label"].astype(str))
    )
    cols = [
        "config_name",
        "exp_name",
        "seed",
        "prediction_file",
        "evaluation_mode",
        "label_col",
        "true_label",
        "pred_label",
        "task2_has_bc_missing",
        "task2_has_ac_nr",
        "task2_has_bc_nr",
        "task2_has_abg_censored",
        "task2_has_abg_borderline",
        "is_chl_mhl_snhl_confusion",
    ]
    existing = [col for col in cols if col in task2.columns]
    return (
        task2.groupby(existing, dropna=False)
        .agg(
            n=("pred_label", "size"),
            errors=("correct", lambda s: int((~s).sum())),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["errors", "n"], ascending=[False, False])
    )


def save_error_analysis(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "prediction_rows_all": output_dir / "prediction_rows_all.csv",
        "error_cases": output_dir / "error_cases.csv",
        "subgroup_metrics": output_dir / "subgroup_metrics.csv",
        "confusion_pairs": output_dir / "confusion_pairs.csv",
        "rule_model_conflicts": output_dir / "rule_model_conflicts.csv",
        "rule_model_conflict_summary": output_dir / "rule_model_conflict_summary.csv",
        "rule_true_model_conflicts": output_dir / "rule_true_model_conflicts.csv",
        "rule_true_model_conflict_summary": output_dir / "rule_true_model_conflict_summary.csv",
        "three_way_conflict_summary": output_dir / "three_way_conflict_summary.csv",
        "task2_clinical_subgroups": output_dir / "task2_clinical_subgroups.csv",
        "task2_confusion_focus": output_dir / "task2_confusion_focus.csv",
        "manifest": output_dir / "error_analysis_manifest.json",
    }

    if df.empty:
        for key, path in files.items():
            if key != "manifest":
                pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "no_prediction_rows_found",
            "files": {key: str(path) for key, path in files.items()},
        }
        files["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    subgroup_df = build_subgroup_metrics(df)
    confusion_df = build_confusion_pairs(df)
    conflict_df = build_rule_model_conflicts(df)
    conflict_summary_df = build_rule_model_conflict_summary(conflict_df)
    rule_true_model_conflict_df = build_rule_true_model_conflicts(df)
    rule_true_model_conflict_summary_df = build_rule_true_model_conflict_summary(df)
    task2_clinical_df = build_task2_clinical_subgroups(df)
    task2_confusion_df = build_task2_confusion_focus(df)
    error_df = df.loc[~df["correct"]].copy()
    error_df = error_df.sort_values(["task", "label_col", "confidence"], ascending=[True, True, False])

    df.to_csv(files["prediction_rows_all"], index=False, encoding="utf-8-sig")
    error_df.to_csv(files["error_cases"], index=False, encoding="utf-8-sig")
    subgroup_df.to_csv(files["subgroup_metrics"], index=False, encoding="utf-8-sig")
    confusion_df.to_csv(files["confusion_pairs"], index=False, encoding="utf-8-sig")
    conflict_df.to_csv(files["rule_model_conflicts"], index=False, encoding="utf-8-sig")
    conflict_summary_df.to_csv(files["rule_model_conflict_summary"], index=False, encoding="utf-8-sig")
    rule_true_model_conflict_df.to_csv(files["rule_true_model_conflicts"], index=False, encoding="utf-8-sig")
    rule_true_model_conflict_summary_df.to_csv(files["rule_true_model_conflict_summary"], index=False, encoding="utf-8-sig")
    rule_true_model_conflict_summary_df.to_csv(files["three_way_conflict_summary"], index=False, encoding="utf-8-sig")
    task2_clinical_df.to_csv(files["task2_clinical_subgroups"], index=False, encoding="utf-8-sig")
    task2_confusion_df.to_csv(files["task2_confusion_focus"], index=False, encoding="utf-8-sig")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "n_prediction_rows": int(len(df)),
        "n_error_rows": int(len(error_df)),
        "n_subgroup_rows": int(len(subgroup_df)),
        "n_rule_model_conflict_rows": int(len(conflict_df)),
        "n_rule_model_conflict_summary_rows": int(len(conflict_summary_df)),
        "n_rule_true_model_conflict_rows": int(len(rule_true_model_conflict_df)),
        "n_rule_true_model_conflict_summary_rows": int(len(rule_true_model_conflict_summary_df)),
        "n_three_way_conflict_summary_rows": int(len(rule_true_model_conflict_summary_df)),
        "n_task2_clinical_subgroup_rows": int(len(task2_clinical_df)),
        "n_task2_confusion_focus_rows": int(len(task2_confusion_df)),
        "files": {key: str(path) for key, path in files.items()},
    }
    files["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Aggregate v7.4 prediction rows into subgroup and error analysis tables.")
    parser.add_argument("--results_dir", default="results_v74", help="Directory containing run outputs.")
    parser.add_argument("--prediction_rows", default="", help="Optional existing prediction_rows_all.csv input.")
    parser.add_argument("--output_dir", default="results_v74/error_analysis", help="Directory for analysis tables.")
    parser.add_argument("--mode", default="both", choices=["configured", "no_support", "locked_test", "both", "all"])
    parser.add_argument("--rule_predictions", default="results_v74/rule_baselines_phase3/rule_baseline_predictions.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    if args.prediction_rows:
        df = add_subgroup_columns(pd.read_csv(args.prediction_rows))
    else:
        df = load_prediction_rows(results_dir, args.mode)
    df = attach_rule_predictions(df, load_rule_predictions(Path(args.rule_predictions)))
    manifest = save_error_analysis(df, output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
