import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from error_analysis_v74 import load_prediction_rows
from preprocessing_v74 import INSUFFICIENT_EVIDENCE_LABEL


DERIVED_RULE_COLUMNS = [
    "forced_pred_label",
    "rule_decision_label",
    "abstain_pred_label",
    "baseline_covered",
    "evidence_status",
    "rule_confidence",
    "warning_reasons",
]
DERIVED_HYBRID_COLUMNS = [
    "hybrid_pred_label",
    "hybrid_used_rule",
    "hybrid_used_model",
    "hybrid_uncertainty_gated_pred_label",
    "hybrid_confidence_gate_used_rule",
    "hybrid_confidence_gate_used_model",
    "hybrid_low_confidence_abstain",
    "hybrid_model_confidence",
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
        "evidence_status",
        "rule_confidence",
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
    use_rule = out["baseline_covered"] & out["rule_confidence"].ge(rule_confidence_threshold)
    out["hybrid_pred_label"] = out["pred_label"]
    out.loc[use_rule, "hybrid_pred_label"] = out.loc[use_rule, "rule_decision_label"]
    out["hybrid_used_rule"] = use_rule
    out["hybrid_used_model"] = ~use_rule

    confidence = get_model_confidence(out)
    threshold = max(0.0, min(1.0, float(model_confidence_threshold or 0.0)))
    low_confidence = (~use_rule) & confidence.notna() & confidence.lt(threshold)
    out["hybrid_uncertainty_gated_pred_label"] = out["hybrid_pred_label"]
    out.loc[low_confidence, "hybrid_uncertainty_gated_pred_label"] = INSUFFICIENT_EVIDENCE_LABEL
    out["hybrid_confidence_gate_used_rule"] = use_rule
    out["hybrid_low_confidence_abstain"] = low_confidence
    out["hybrid_confidence_gate_used_model"] = (~use_rule) & (~low_confidence)
    out["hybrid_model_confidence"] = confidence.astype(float)
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
    ]
    output_cols = [col for col in output_cols if col in pivot.columns]
    return pivot[output_cols]


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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir = Path(args.paper_tables_dir)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    locked_mode = args.mode == "locked_test"
    pred_path = out_dir / ("hybrid_predictions_locked_test.csv" if locked_mode else "hybrid_predictions.csv")
    summary_path = out_dir / ("hybrid_locked_test_summary.csv" if locked_mode else "hybrid_summary.csv")
    main_summary_path = paper_tables_dir / (
        "main_hybrid_summary_locked_test.csv" if locked_mode else "main_hybrid_summary.csv"
    )
    manifest_path = out_dir / "hybrid_manifest.json"
    merged.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    main_summary.to_csv(main_summary_path, index=False, encoding="utf-8-sig")
    manifest = {
        "results_dir": args.results_dir,
        "mode": args.mode,
        "model_predictions_file": args.model_predictions_file,
        "rule_predictions": args.rule_predictions,
        "rule_confidence_threshold": args.rule_confidence_threshold,
        "model_confidence_threshold": args.model_confidence_threshold,
        "outputs": {
            "predictions": str(pred_path),
            "summary": str(summary_path),
            "main_summary": str(main_summary_path),
            "locked_summary": str(summary_path) if locked_mode else None,
            "locked_main_summary": str(main_summary_path) if locked_mode else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare model-only, rule-only, and rule+model hybrid outputs.")
    parser.add_argument("--results_dir", default="results_v74")
    parser.add_argument("--rule_predictions", default="results_v74/rule_baselines_phase3/rule_baseline_predictions.csv")
    parser.add_argument("--model_predictions_file", default="", help="Optional existing model or hybrid prediction CSV to recompute rule/hybrid columns.")
    parser.add_argument("--output_dir", default="paper_v74/hybrid_evaluation")
    parser.add_argument("--paper_tables_dir", default="paper_v74/tables")
    parser.add_argument("--mode", default="no_support", choices=["configured", "no_support", "locked_test", "both", "all"])
    parser.add_argument("--rule_confidence_threshold", type=float, default=0.8)
    parser.add_argument("--model_confidence_threshold", type=float, default=0.6)
    args = parser.parse_args()
    summary = run(args)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
