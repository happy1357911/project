import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from clinical_rules_v74 import has_hard_rule_warning
from error_analysis_v74 import load_prediction_rows
from preprocessing_v74 import INSUFFICIENT_EVIDENCE_LABEL


DERIVED_RULE_COLUMNS = [
    "forced_pred_label",
    "rule_decision_label",
    "abstain_pred_label",
    "baseline_covered",
    "complete_for_rule",
    "evidence_status",
    "rule_confidence",
    "rule_evidence_score",
    "score_deductions",
    "warning_reasons",
    "hard_warning_flags",
]
DERIVED_HYBRID_COLUMNS = [
    "hybrid_pred_label",
    "hybrid_used_rule",
    "hybrid_used_model",
    "hybrid_uncertainty_gated_pred_label",
    "hybrid_confidence_gate_used_rule",
    "hybrid_confidence_gate_used_model",
    "hybrid_low_confidence_abstain",
    "hybrid_hard_warning_blocked_rule",
    "hybrid_model_confidence",
    "hybrid_decision_reason",
    "hybrid_warning_reasons",
]


def metric_dict(y_true, y_pred):
    y_true = [str(value) for value in y_true]
    y_pred = [str(value) for value in y_pred]
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": int(len(y_true)),
        "labels": ",".join(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(safe_balanced_accuracy(y_true, y_pred)),
    }


def safe_balanced_accuracy(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return balanced_accuracy_score(y_true, y_pred)


def load_rule_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Rule baseline predictions not found: {path}")
    df = pd.read_csv(path)
    keep_cols = [
        "task",
        "case_id",
        "ear_side",
        "true_label",
        "forced_pred_label",
        "rule_decision_label",
        "abstain_pred_label",
        "baseline_covered",
        "complete_for_rule",
        "evidence_status",
        "rule_confidence",
        "rule_evidence_score",
        "score_deductions",
        "warning_reasons",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    return df[keep_cols].copy()


def load_model_prediction_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model prediction file not found: {path}")
    df = pd.read_csv(path)
    derived_cols = [col for col in DERIVED_RULE_COLUMNS + DERIVED_HYBRID_COLUMNS if col in df.columns]
    if derived_cols:
        df = df.drop(columns=derived_cols)
    return df


def merge_model_rule_predictions(model_df: pd.DataFrame, rule_df: pd.DataFrame) -> pd.DataFrame:
    model = model_df.copy()
    rule = rule_df.copy()
    for col in ["case_id", "ear_side", "task", "true_label"]:
        model[col] = model[col].astype(str)
        rule[col] = rule[col].astype(str)
    merged = model.merge(
        rule,
        on=["task", "case_id", "ear_side", "true_label"],
        how="left",
        suffixes=("", "_rule"),
    )
    merged["baseline_covered"] = merged["baseline_covered"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "complete_for_rule" not in merged.columns:
        merged["complete_for_rule"] = merged["baseline_covered"]
    else:
        complete_raw = merged["complete_for_rule"]
        complete_mask = complete_raw.astype(str).str.lower().isin({"true", "1", "yes"})
        complete_mask = complete_mask.mask(complete_raw.isna(), merged["baseline_covered"])
        merged["complete_for_rule"] = complete_mask.astype(bool)
    merged["rule_confidence"] = pd.to_numeric(merged["rule_confidence"], errors="coerce").fillna(0.0)
    merged["forced_pred_label"] = merged["forced_pred_label"].fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
    if "rule_decision_label" not in merged.columns:
        merged["rule_decision_label"] = merged["forced_pred_label"]
    merged["rule_decision_label"] = merged["rule_decision_label"].fillna(merged["forced_pred_label"]).astype(str)
    merged["abstain_pred_label"] = merged["abstain_pred_label"].fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
    merged["pred_label"] = merged["pred_label"].astype(str)
    return merged


def get_model_confidence(out: pd.DataFrame) -> pd.Series:
    for col in ["confidence", "model_confidence", "pred_confidence", "max_probability"]:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
    return pd.Series([pd.NA] * len(out), index=out.index, dtype="Float64")


def add_hybrid_predictions(
    df: pd.DataFrame,
    rule_confidence_threshold: float,
    model_confidence_threshold: float = 0.6,
) -> pd.DataFrame:
    out = df.copy()
    rule_label = out["rule_decision_label"].fillna("").astype(str).str.strip()
    rule_label_available = (
        rule_label.ne("")
        & rule_label.str.lower().ne("none")
        & rule_label.ne(INSUFFICIENT_EVIDENCE_LABEL)
    )
    warning_text = out.get("warning_reasons", pd.Series("", index=out.index)).fillna("").astype(str).str.strip(";")
    hard_warning = warning_text.map(has_hard_rule_warning)
    baseline_covered = out.get("baseline_covered", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    complete_for_rule = out.get("complete_for_rule", baseline_covered).fillna(False).astype(bool)
    use_rule = (
        rule_label_available
        & baseline_covered
        & complete_for_rule
        & out["rule_confidence"].ge(rule_confidence_threshold)
        & (~hard_warning)
    )
    out["hybrid_pred_label"] = out["pred_label"]
    out.loc[use_rule, "hybrid_pred_label"] = out.loc[use_rule, "rule_decision_label"]
    out["hybrid_used_rule"] = use_rule
    out["hybrid_used_model"] = ~use_rule
    out["hybrid_hard_warning_blocked_rule"] = rule_label_available & hard_warning & (~use_rule)

    confidence = get_model_confidence(out)
    threshold = max(0.0, min(1.0, float(model_confidence_threshold or 0.0)))
    low_confidence = (~use_rule) & confidence.notna() & confidence.lt(threshold)
    out["hybrid_uncertainty_gated_pred_label"] = out["hybrid_pred_label"]
    out.loc[low_confidence, "hybrid_uncertainty_gated_pred_label"] = INSUFFICIENT_EVIDENCE_LABEL
    out["hybrid_confidence_gate_used_rule"] = use_rule
    out["hybrid_low_confidence_abstain"] = low_confidence
    out["hybrid_confidence_gate_used_model"] = (~use_rule) & (~low_confidence)
    out["hybrid_model_confidence"] = confidence.astype(float)

    reason = pd.Series("model_fallback_low_rule_score", index=out.index, dtype=object)
    reason = reason.mask(~rule_label_available, "model_fallback_no_rule_prediction")
    reason = reason.mask(rule_label_available & (~baseline_covered), "model_fallback_rule_not_covered")
    reason = reason.mask(rule_label_available & baseline_covered & (~complete_for_rule), "model_fallback_incomplete_rule_data")
    reason = reason.mask(out["hybrid_hard_warning_blocked_rule"], "model_fallback_hard_rule_warning")
    reason = reason.mask(use_rule & warning_text.ne(""), "rule_score_ge_threshold_with_warning")
    reason = reason.mask(use_rule & warning_text.eq(""), "rule_score_ge_threshold")
    reason = reason.mask(low_confidence, "abstain_low_model_confidence")
    out["hybrid_decision_reason"] = reason

    warning_base = out.get("warning_reasons", pd.Series("", index=out.index)).fillna("").astype(str)
    out["hybrid_warning_reasons"] = warning_base
    out.loc[low_confidence, "hybrid_warning_reasons"] = (
        out.loc[low_confidence, "hybrid_warning_reasons"].str.strip(";")
        + ";low_model_confidence"
    ).str.strip(";")
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    strategies = {
        "model_only": "pred_label",
        "rule_decision": "rule_decision_label",
        "rule_forced": "forced_pred_label",
        "rule_abstain_as_error": "abstain_pred_label",
        "hybrid_rule_first": "hybrid_pred_label",
        "hybrid_rule_first_confidence_gate": "hybrid_uncertainty_gated_pred_label",
    }
    rows = []
    group_cols = [
        "config_name",
        "exp_name",
        "seed",
        "prediction_file",
        "evaluation_mode",
        "task",
        "label_col",
    ]
    existing = [col for col in group_cols if col in df.columns]
    for keys, sub in df.groupby(existing, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: value for col, value in zip(existing, keys)}
        for strategy, pred_col in strategies.items():
            row = dict(base)
            row["strategy"] = strategy
            row.update(metric_dict(sub["true_label"], sub[pred_col]))
            if strategy == "hybrid_rule_first":
                row["hybrid_rule_rate"] = float(sub["hybrid_used_rule"].mean())
                row["hybrid_model_rate"] = float(sub["hybrid_used_model"].mean())
                row["low_confidence_abstain_rate"] = 0.0
            elif strategy == "hybrid_rule_first_confidence_gate":
                row["hybrid_rule_rate"] = float(sub["hybrid_confidence_gate_used_rule"].mean())
                row["hybrid_model_rate"] = float(sub["hybrid_confidence_gate_used_model"].mean())
                row["low_confidence_abstain_rate"] = float(sub["hybrid_low_confidence_abstain"].mean())
            else:
                row["hybrid_rule_rate"] = None
                row["hybrid_model_rate"] = None
                row["low_confidence_abstain_rate"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def iter_groups(df: pd.DataFrame, group_cols: list[str]):
    if group_cols:
        for keys, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            yield {col: value for col, value in zip(group_cols, keys)}, sub
    else:
        yield {}, df


def non_empty_text_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def available_label_mask(series: pd.Series) -> pd.Series:
    labels = series.fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str).str.strip()
    return labels.ne("") & labels.ne(INSUFFICIENT_EVIDENCE_LABEL) & labels.str.lower().ne("nan")


def _safe_rate(mask: pd.Series, denom: pd.Series | None = None) -> float:
    if denom is None:
        return float(mask.mean()) if len(mask) else math.nan
    denom = denom.fillna(False).astype(bool)
    if int(denom.sum()) == 0:
        return math.nan
    return float(mask[denom].mean())


def compute_clinical_value_metrics(sub: pd.DataFrame, strategy: str = "hybrid_rule_first") -> dict:
    true_label = sub["true_label"].astype(str)
    model_pred = sub["pred_label"].astype(str)
    rule_pred = sub.get("forced_pred_label", pd.Series(INSUFFICIENT_EVIDENCE_LABEL, index=sub.index)).fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
    rule_available = available_label_mask(rule_pred)
    model_correct = model_pred.eq(true_label)
    rule_correct = rule_available & rule_pred.eq(true_label)
    rule_wrong = rule_available & (~rule_correct)
    rule_model_conflict = rule_available & rule_pred.ne(model_pred)
    rule_correction = rule_correct & (~model_correct)
    warning_source = sub.get("hybrid_warning_reasons", sub.get("warning_reasons", pd.Series("", index=sub.index)))
    warning_mask = non_empty_text_mask(warning_source)

    if strategy == "hybrid_rule_first_confidence_gate":
        fallback_mask = sub.get("hybrid_confidence_gate_used_model", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        abstain_mask = sub.get("hybrid_low_confidence_abstain", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
    else:
        fallback_mask = sub.get("hybrid_used_model", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        abstain_mask = pd.Series(False, index=sub.index)

    return {
        "rule_available_rate": _safe_rate(rule_available),
        "rule_correct_when_available_rate": _safe_rate(rule_correct, rule_available),
        "rule_failure_rate": _safe_rate(rule_wrong, rule_available),
        "rule_correction_rate": _safe_rate(rule_correction),
        "model_fallback_success_rate": _safe_rate(fallback_mask & model_correct, fallback_mask),
        "rule_model_conflict_rate": _safe_rate(rule_model_conflict),
        "warning_rate": _safe_rate(warning_mask | abstain_mask),
    }


def parse_threshold_list(value: str) -> list[float]:
    values = []
    for part in str(value).split(','):
        part = part.strip()
        if part:
            values.append(float(part))
    return values or [0.8]


def build_threshold_sweep(base_merged: pd.DataFrame, rule_thresholds: list[float], model_thresholds: list[float]) -> pd.DataFrame:
    group_cols = [col for col in ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode"] if col in base_merged.columns]
    rows = []
    for rule_threshold in rule_thresholds:
        for model_threshold in model_thresholds:
            merged = add_hybrid_predictions(
                base_merged,
                rule_confidence_threshold=rule_threshold,
                model_confidence_threshold=model_threshold,
            )
            for base, sub in iter_groups(merged, group_cols):
                row = {
                    **base,
                    "rule_confidence_threshold": float(rule_threshold),
                    "model_confidence_threshold": float(model_threshold),
                    "strategy": "hybrid_rule_first_confidence_gate",
                }
                task_metrics = []
                for task_name, task_sub in sub.groupby("task", dropna=False):
                    metrics = metric_dict(task_sub["true_label"], task_sub["hybrid_uncertainty_gated_pred_label"])
                    row[f"{task_name}_macro_f1"] = metrics["macro_f1"]
                    row[f"{task_name}_accuracy"] = metrics["accuracy"]
                    row[f"{task_name}_balanced_accuracy"] = metrics["balanced_accuracy"]
                    task_metrics.append(metrics["macro_f1"])
                row["mean_macro_f1"] = float(np.nanmean(task_metrics)) if task_metrics else math.nan
                row.update(compute_clinical_value_metrics(sub, "hybrid_rule_first_confidence_gate"))
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in group_cols if col in out.columns] + ["rule_confidence_threshold", "model_confidence_threshold"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def mcnemar_p_value(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    statistic = (abs(b - c) - 1.0) ** 2 / discordant
    return float(math.erfc(math.sqrt(max(statistic, 0.0) / 2.0)))


def build_strategy_mcnemar_tests(merged: pd.DataFrame) -> pd.DataFrame:
    strategies = {
        "model_only": "pred_label",
        "rule_forced": "forced_pred_label",
        "rule_abstain_as_error": "abstain_pred_label",
        "hybrid_rule_first": "hybrid_pred_label",
        "hybrid_rule_first_confidence_gate": "hybrid_uncertainty_gated_pred_label",
    }
    pairs = [
        ("model_only", "hybrid_rule_first"),
        ("model_only", "hybrid_rule_first_confidence_gate"),
        ("rule_forced", "hybrid_rule_first"),
        ("rule_abstain_as_error", "hybrid_rule_first_confidence_gate"),
    ]
    group_cols = [col for col in ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "task", "label_col"] if col in merged.columns]
    rows = []
    for base, sub in iter_groups(merged, group_cols):
        true_label = sub["true_label"].astype(str)
        for left_name, right_name in pairs:
            left_correct = sub[strategies[left_name]].astype(str).eq(true_label)
            right_correct = sub[strategies[right_name]].astype(str).eq(true_label)
            b = int((left_correct & (~right_correct)).sum())
            c = int(((~left_correct) & right_correct).sum())
            rows.append({
                **base,
                "baseline_strategy": left_name,
                "comparison_strategy": right_name,
                "n": int(len(sub)),
                "baseline_accuracy": float(left_correct.mean()) if len(sub) else math.nan,
                "comparison_accuracy": float(right_correct.mean()) if len(sub) else math.nan,
                "paired_accuracy_delta": float(right_correct.mean() - left_correct.mean()) if len(sub) else math.nan,
                "baseline_correct_comparison_wrong": b,
                "baseline_wrong_comparison_correct": c,
                "mcnemar_p_value": mcnemar_p_value(b, c),
            })
    return pd.DataFrame(rows)


def build_main_hybrid_summary(summary: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        col
        for col in ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode"]
        if col in summary.columns
    ]
    index_cols = group_cols + ["strategy"]
    pivot = summary.pivot_table(
        index=index_cols,
        columns="task",
        values="macro_f1",
        aggfunc="mean",
    ).reset_index()

    task_order = ["Task1", "Task2", "Task3"]
    observed_tasks = [task for task in task_order if task in pivot.columns]
    observed_tasks.extend(sorted(task for task in pivot.columns if str(task).startswith("Task") and task not in observed_tasks))
    task_metric_cols = []
    for task in observed_tasks:
        metric_col = f"{task}_macro_f1"
        pivot = pivot.rename(columns={task: metric_col})
        task_metric_cols.append(metric_col)
    pivot["mean_macro_f1"] = pivot[task_metric_cols].mean(axis=1, skipna=True) if task_metric_cols else None

    strategy_order = [
        "rule_decision",
        "rule_forced",
        "rule_abstain_as_error",
        "model_only",
        "hybrid_rule_first",
        "hybrid_rule_first_confidence_gate",
    ]
    rate_rows = []
    rate_group_cols = [col for col in group_cols if col in merged.columns]
    for base, sub in iter_groups(merged, rate_group_cols):
        rule_coverage = float(sub["baseline_covered"].mean())
        abstain_rate = float(sub["abstain_pred_label"].eq(INSUFFICIENT_EVIDENCE_LABEL).mean())
        hybrid_rule_rate = float(sub["hybrid_used_rule"].mean())
        hybrid_model_rate = float(sub["hybrid_used_model"].mean())
        gated_rule_rate = float(sub["hybrid_confidence_gate_used_rule"].mean())
        gated_model_rate = float(sub["hybrid_confidence_gate_used_model"].mean())
        gated_abstain_rate = float(sub["hybrid_uncertainty_gated_pred_label"].eq(INSUFFICIENT_EVIDENCE_LABEL).mean())
        low_confidence_abstain_rate = float(sub["hybrid_low_confidence_abstain"].mean())
        for strategy in strategy_order:
            row = dict(base)
            row["strategy"] = strategy
            row["coverage"] = None if strategy == "model_only" else rule_coverage
            if strategy == "rule_abstain_as_error":
                row["abstain_rate"] = abstain_rate
            elif strategy == "hybrid_rule_first_confidence_gate":
                row["abstain_rate"] = gated_abstain_rate
            else:
                row["abstain_rate"] = 0.0
            row["hybrid_rule_rate"] = hybrid_rule_rate if strategy == "hybrid_rule_first" else None
            row["hybrid_model_rate"] = hybrid_model_rate if strategy == "hybrid_rule_first" else None
            row["low_confidence_abstain_rate"] = None
            if strategy == "hybrid_rule_first_confidence_gate":
                row["hybrid_rule_rate"] = gated_rule_rate
                row["hybrid_model_rate"] = gated_model_rate
                row["low_confidence_abstain_rate"] = low_confidence_abstain_rate
            row.update(compute_clinical_value_metrics(sub, strategy))
            rate_rows.append(row)

    rates = pd.DataFrame(rate_rows)
    if not rates.empty:
        merge_cols = [col for col in index_cols if col in rates.columns]
        pivot = pivot.merge(rates, on=merge_cols, how="left", suffixes=("", "_overall"))

    pivot["strategy_order"] = pivot["strategy"].map({name: idx for idx, name in enumerate(strategy_order)})
    sort_cols = [col for col in group_cols if col in pivot.columns] + ["strategy_order"]
    if sort_cols:
        pivot = pivot.sort_values(sort_cols).reset_index(drop=True)
    pivot = pivot.drop(columns=["strategy_order"])

    output_cols = index_cols + task_metric_cols + [
        "mean_macro_f1",
        "coverage",
        "abstain_rate",
        "hybrid_rule_rate",
        "hybrid_model_rate",
        "low_confidence_abstain_rate",
        "rule_available_rate",
        "rule_correct_when_available_rate",
        "rule_failure_rate",
        "rule_correction_rate",
        "model_fallback_success_rate",
        "rule_model_conflict_rate",
        "warning_rate",
    ]
    output_cols = [col for col in output_cols if col in pivot.columns]
    return pivot[output_cols]



def build_method_comparison_summary(main_summary: pd.DataFrame) -> pd.DataFrame:
    if main_summary.empty or "strategy" not in main_summary.columns:
        return pd.DataFrame()
    group_cols = [col for col in ["evaluation_mode", "strategy"] if col in main_summary.columns]
    if "strategy" not in group_cols:
        group_cols.append("strategy")
    metric_cols = [
        col for col in [
            "Task1_macro_f1", "Task2_macro_f1", "Task3_macro_f1", "mean_macro_f1",
            "coverage", "abstain_rate", "hybrid_rule_rate", "hybrid_model_rate",
            "low_confidence_abstain_rate", "rule_available_rate",
            "rule_correct_when_available_rate", "rule_failure_rate", "rule_correction_rate",
            "model_fallback_success_rate", "rule_model_conflict_rate", "warning_rate",
        ] if col in main_summary.columns
    ]
    rows = []
    for key, sub in main_summary.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row["n_profiles"] = int(len(sub))
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}__mean"] = float(values.mean()) if values.notna().any() else math.nan
            row[f"{col}__std"] = float(values.std(ddof=0)) if values.notna().any() else math.nan
            row[f"{col}__min"] = float(values.min()) if values.notna().any() else math.nan
            row[f"{col}__max"] = float(values.max()) if values.notna().any() else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    strategy_order = {
        "rule_decision": 0,
        "rule_forced": 1,
        "rule_abstain_as_error": 2,
        "model_only": 3,
        "hybrid_rule_first": 4,
        "hybrid_rule_first_confidence_gate": 5,
    }
    out["strategy_order"] = out["strategy"].map(strategy_order).fillna(99)
    sort_cols = [col for col in ["evaluation_mode", "strategy_order"] if col in out.columns]
    out = out.sort_values(sort_cols).drop(columns=["strategy_order"]).reset_index(drop=True)
    return out
def build_rule_contribution_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()
    group_cols = [
        col
        for col in ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "task", "label_col"]
        if col in merged.columns
    ]
    rows = []
    for base, sub in iter_groups(merged, group_cols):
        rule_available = available_label_mask(sub.get("forced_pred_label", pd.Series("", index=sub.index)))
        rule_covered = sub.get("baseline_covered", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        abstain_pred = sub.get(
            "abstain_pred_label",
            pd.Series(INSUFFICIENT_EVIDENCE_LABEL, index=sub.index),
        ).fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
        warning_mask = non_empty_text_mask(sub.get("warning_reasons", pd.Series("", index=sub.index)))
        true_label = sub["true_label"].astype(str)
        forced_pred = sub["forced_pred_label"].fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
        row = {
            **base,
            "n": int(len(sub)),
            "rule_available_rate": _safe_rate(rule_available),
            "rule_coverage_rate": _safe_rate(rule_covered),
            "rule_abstain_rate": _safe_rate(abstain_pred.eq(INSUFFICIENT_EVIDENCE_LABEL)),
            "warning_rate": _safe_rate(warning_mask),
            "mean_rule_confidence": float(pd.to_numeric(sub.get("rule_confidence", pd.Series(np.nan, index=sub.index)), errors="coerce").mean()),
            "mean_rule_evidence_score": float(pd.to_numeric(sub.get("rule_evidence_score", pd.Series(np.nan, index=sub.index)), errors="coerce").mean()),
        }
        row.update({
            f"rule_forced_{key}": value
            for key, value in metric_dict(true_label, forced_pred).items()
            if key != "labels"
        })
        row.update({
            f"rule_abstain_as_error_{key}": value
            for key, value in metric_dict(true_label, abstain_pred).items()
            if key != "labels"
        })
        if bool(rule_covered.any()):
            covered_metrics = metric_dict(true_label[rule_covered], forced_pred[rule_covered])
            row.update({
                f"rule_covered_only_{key}": value
                for key, value in covered_metrics.items()
                if key != "labels"
            })
        else:
            row.update({
                "rule_covered_only_n": 0,
                "rule_covered_only_accuracy": math.nan,
                "rule_covered_only_macro_f1": math.nan,
                "rule_covered_only_balanced_accuracy": math.nan,
            })
        row.update(compute_clinical_value_metrics(sub, "hybrid_rule_first"))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in group_cols if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_hybrid_explainability_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty or "hybrid_decision_reason" not in merged.columns:
        return pd.DataFrame()
    group_cols = [
        col
        for col in [
            "config_name", "exp_name", "seed", "prediction_file", "evaluation_mode",
            "task", "label_col", "hybrid_decision_reason",
        ]
        if col in merged.columns
    ]
    rows = []
    for base, sub in iter_groups(merged, group_cols):
        true_label = sub["true_label"].astype(str)
        hybrid_pred = sub["hybrid_pred_label"].fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
        model_pred = sub["pred_label"].astype(str)
        rule_pred = sub.get("forced_pred_label", pd.Series(INSUFFICIENT_EVIDENCE_LABEL, index=sub.index)).fillna(INSUFFICIENT_EVIDENCE_LABEL).astype(str)
        rule_available = available_label_mask(rule_pred)
        warning_mask = non_empty_text_mask(sub.get("hybrid_warning_reasons", sub.get("warning_reasons", pd.Series("", index=sub.index))))
        metrics = metric_dict(true_label, hybrid_pred)
        row = {
            **base,
            "n": int(len(sub)),
            "hybrid_accuracy": metrics["accuracy"],
            "hybrid_macro_f1": metrics["macro_f1"],
            "hybrid_balanced_accuracy": metrics["balanced_accuracy"],
            "model_correct_rate": float(model_pred.eq(true_label).mean()) if len(sub) else math.nan,
            "rule_correct_when_available_rate": _safe_rate(rule_available & rule_pred.eq(true_label), rule_available),
            "hybrid_used_rule_rate": float(sub.get("hybrid_used_rule", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "hybrid_used_model_rate": float(sub.get("hybrid_used_model", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "low_confidence_abstain_rate": float(sub.get("hybrid_low_confidence_abstain", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "rule_model_conflict_rate": _safe_rate(rule_available & rule_pred.ne(model_pred)),
            "warning_rate": _safe_rate(warning_mask),
            "mean_rule_confidence": float(pd.to_numeric(sub.get("rule_confidence", pd.Series(np.nan, index=sub.index)), errors="coerce").mean()),
            "mean_rule_evidence_score": float(pd.to_numeric(sub.get("rule_evidence_score", pd.Series(np.nan, index=sub.index)), errors="coerce").mean()),
            "mean_model_confidence": float(pd.to_numeric(sub.get("hybrid_model_confidence", pd.Series(np.nan, index=sub.index)), errors="coerce").mean()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    total_cols = [col for col in group_cols if col != "hybrid_decision_reason"]
    out["reason_fraction"] = out["n"] / out.groupby(total_cols, dropna=False)["n"].transform("sum")
    return out.sort_values(total_cols + ["n"], ascending=[True] * len(total_cols) + [False]).reset_index(drop=True)
def build_decision_reason_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty or "hybrid_decision_reason" not in merged.columns:
        return pd.DataFrame()
    group_cols = [
        col
        for col in ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "task", "label_col", "hybrid_decision_reason"]
        if col in merged.columns
    ]
    summary = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            n=("hybrid_decision_reason", "size"),
            mean_model_confidence=("hybrid_model_confidence", "mean"),
            rule_rate=("hybrid_confidence_gate_used_rule", "mean"),
            model_rate=("hybrid_confidence_gate_used_model", "mean"),
            abstain_rate=("hybrid_low_confidence_abstain", "mean"),
        )
        .reset_index()
    )
    total_cols = [col for col in group_cols if col != "hybrid_decision_reason"]
    totals = summary.groupby(total_cols, dropna=False)["n"].transform("sum") if total_cols else summary["n"].sum()
    summary["reason_fraction"] = summary["n"] / totals
    return summary.sort_values(total_cols + ["n"], ascending=[True] * len(total_cols) + [False]).reset_index(drop=True)


def run(args):
    if args.model_predictions_file:
        model_df = load_model_prediction_file(Path(args.model_predictions_file))
    else:
        model_df = load_prediction_rows(Path(args.results_dir), args.mode)
    if model_df.empty:
        raise FileNotFoundError(f"No model prediction rows found under {args.results_dir} with mode={args.mode}")
    rule_df = load_rule_predictions(Path(args.rule_predictions))
    merged = merge_model_rule_predictions(model_df, rule_df)
    merged = add_hybrid_predictions(
        merged,
        args.rule_confidence_threshold,
        model_confidence_threshold=args.model_confidence_threshold,
    )
    summary = summarize(merged)
    main_summary = build_main_hybrid_summary(summary, merged)
    reason_summary = build_decision_reason_summary(merged)
    rule_contribution_summary = build_rule_contribution_summary(merged)
    hybrid_explainability_summary = build_hybrid_explainability_summary(merged)
    method_comparison_summary = build_method_comparison_summary(main_summary)
    threshold_sweep = build_threshold_sweep(
        merged,
        parse_threshold_list(args.rule_thresholds),
        parse_threshold_list(args.model_thresholds),
    ) if args.threshold_sweep else pd.DataFrame()
    mcnemar_tests = build_strategy_mcnemar_tests(merged)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir = Path(args.paper_tables_dir)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    locked_mode = args.mode == "locked_test"
    pred_path = out_dir / ("hybrid_predictions_locked_test.csv" if locked_mode else "hybrid_predictions.csv")
    summary_path = out_dir / ("hybrid_locked_test_summary.csv" if locked_mode else "hybrid_summary.csv")
    reason_summary_path = out_dir / (
        "hybrid_decision_reason_summary_locked_test.csv" if locked_mode else "hybrid_decision_reason_summary.csv"
    )
    main_summary_path = paper_tables_dir / (
        "main_hybrid_summary_locked_test.csv" if locked_mode else "main_hybrid_summary.csv"
    )
    rule_contribution_path = paper_tables_dir / (
        "rule_contribution_summary_locked_test.csv" if locked_mode else "rule_contribution_summary.csv"
    )
    hybrid_explainability_path = paper_tables_dir / (
        "hybrid_explainability_summary_locked_test.csv" if locked_mode else "hybrid_explainability_summary.csv"
    )
    method_comparison_path = paper_tables_dir / (
        "main_method_comparison_locked_test.csv" if locked_mode else "main_method_comparison_no_support.csv"
    )
    threshold_sweep_path = out_dir / (
        "hybrid_threshold_sweep_locked_test.csv" if locked_mode else "hybrid_threshold_sweep.csv"
    )
    mcnemar_path = out_dir / (
        "hybrid_strategy_mcnemar_locked_test.csv" if locked_mode else "hybrid_strategy_mcnemar.csv"
    )
    manifest_path = out_dir / "hybrid_manifest.json"
    merged.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    reason_summary.to_csv(reason_summary_path, index=False, encoding="utf-8-sig")
    main_summary.to_csv(main_summary_path, index=False, encoding="utf-8-sig")
    rule_contribution_summary.to_csv(rule_contribution_path, index=False, encoding="utf-8-sig")
    hybrid_explainability_summary.to_csv(hybrid_explainability_path, index=False, encoding="utf-8-sig")
    method_comparison_summary.to_csv(method_comparison_path, index=False, encoding="utf-8-sig")
    mcnemar_tests.to_csv(mcnemar_path, index=False, encoding="utf-8-sig")
    if not threshold_sweep.empty:
        threshold_sweep.to_csv(threshold_sweep_path, index=False, encoding="utf-8-sig")
    manifest = {
        "results_dir": args.results_dir,
        "mode": args.mode,
        "model_predictions_file": args.model_predictions_file,
        "rule_predictions": args.rule_predictions,
        "rule_confidence_threshold": args.rule_confidence_threshold,
        "model_confidence_threshold": args.model_confidence_threshold,
        "threshold_sweep_enabled": bool(args.threshold_sweep),
        "rule_thresholds": args.rule_thresholds,
        "model_thresholds": args.model_thresholds,
        "outputs": {
            "predictions": str(pred_path),
            "summary": str(summary_path),
            "decision_reason_summary": str(reason_summary_path),
            "main_summary": str(main_summary_path),
            "rule_contribution_summary": str(rule_contribution_path),
            "hybrid_explainability_summary": str(hybrid_explainability_path),
            "method_comparison_summary": str(method_comparison_path),
            "threshold_sweep": str(threshold_sweep_path) if args.threshold_sweep else None,
            "strategy_mcnemar": str(mcnemar_path),
            "locked_summary": str(summary_path) if locked_mode else None,
            "locked_main_summary": str(main_summary_path) if locked_mode else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare neural, clinical-rule, and hybrid decision-support outputs.")
    parser.add_argument("--results_dir", default="results_v74")
    parser.add_argument("--rule_predictions", default="results_v74/rule_baselines_phase3/rule_baseline_predictions.csv")
    parser.add_argument("--model_predictions_file", default="", help="Optional existing model or hybrid prediction CSV to recompute rule/hybrid columns.")
    parser.add_argument("--output_dir", default="paper_v74/hybrid_evaluation")
    parser.add_argument("--paper_tables_dir", default="paper_v74/tables")
    parser.add_argument("--mode", default="no_support", choices=["configured", "no_support", "locked_test", "both", "all"])
    parser.add_argument("--rule_confidence_threshold", type=float, default=0.8)
    parser.add_argument("--model_confidence_threshold", type=float, default=0.6)
    parser.add_argument("--threshold_sweep", action="store_true", help="Export a grid search over rule/model confidence thresholds.")
    parser.add_argument("--rule_thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--model_thresholds", default="0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()
    summary = run(args)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
