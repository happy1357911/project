import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from error_analysis_v74 import load_prediction_rows


GROUP_COLS = ["config_name", "exp_name", "seed", "prediction_file", "evaluation_mode", "task", "label_col"]
REQUIRED_PREDICTION_COLUMNS = ["true_label", "pred_label", "confidence"]


def parse_thresholds(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_prob_json(value) -> Dict[str, float]:
    if pd.isna(value):
        return {}
    try:
        obj = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for key, prob in obj.items():
        try:
            out[str(key)] = float(prob)
        except (TypeError, ValueError):
            continue
    return out


def multiclass_brier_score(true_label: str, probs: Dict[str, float]) -> float:
    labels = set(probs.keys()) | {str(true_label)}
    if not labels:
        return np.nan
    score = 0.0
    for label in labels:
        target = 1.0 if label == str(true_label) else 0.0
        score += (float(probs.get(label, 0.0)) - target) ** 2
    return float(score)


def add_calibration_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "confidence" not in out.columns and "model_confidence" in out.columns:
        out["confidence"] = out["model_confidence"]
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["correct_bool"] = out["true_label"].astype(str).eq(out["pred_label"].astype(str))
    if "correct" in out.columns:
        parsed_correct = out["correct"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        out["correct_bool"] = parsed_correct
    if "prob_json" in out.columns:
        probs = out["prob_json"].apply(parse_prob_json)
        out["brier_score"] = [
            multiclass_brier_score(true, prob)
            for true, prob in zip(out["true_label"].astype(str), probs)
        ]
    else:
        out["brier_score"] = np.nan
    return out


def iter_groups(df: pd.DataFrame, group_cols: Iterable[str]):
    existing = [col for col in group_cols if col in df.columns]
    if existing:
        for keys, sub in df.groupby(existing, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            yield {col: value for col, value in zip(existing, keys)}, sub
    else:
        yield {}, df


def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base, sub in iter_groups(df, GROUP_COLS):
        sub = sub.dropna(subset=["confidence"])
        if sub.empty:
            continue
        row = dict(base)
        row["n"] = int(len(sub))
        row["accuracy"] = float(sub["correct_bool"].mean())
        row["mean_confidence"] = float(sub["confidence"].mean())
        row["mean_brier_score"] = float(sub["brier_score"].mean()) if sub["brier_score"].notna().any() else np.nan
        row["confidence_accuracy_gap"] = float(row["mean_confidence"] - row["accuracy"])
        rows.append(row)
    return pd.DataFrame(rows)


def ece_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for base, sub in iter_groups(df, GROUP_COLS):
        sub = sub.dropna(subset=["confidence"])
        if sub.empty:
            continue
        total = len(sub)
        for idx in range(n_bins):
            left = bins[idx]
            right = bins[idx + 1]
            if idx == n_bins - 1:
                mask = sub["confidence"].between(left, right, inclusive="both")
            else:
                mask = (sub["confidence"] >= left) & (sub["confidence"] < right)
            bin_df = sub.loc[mask]
            if bin_df.empty:
                continue
            acc = float(bin_df["correct_bool"].mean())
            conf = float(bin_df["confidence"].mean())
            row = dict(base)
            row.update({
                "bin_index": int(idx),
                "bin_left": float(left),
                "bin_right": float(right),
                "n": int(len(bin_df)),
                "fraction": float(len(bin_df) / total),
                "accuracy": acc,
                "mean_confidence": conf,
                "abs_gap": float(abs(acc - conf)),
                "ece_contribution": float((len(bin_df) / total) * abs(acc - conf)),
            })
            rows.append(row)
    bins_df = pd.DataFrame(rows)
    if bins_df.empty:
        return bins_df
    ece = (
        bins_df.groupby([col for col in GROUP_COLS if col in bins_df.columns], dropna=False)["ece_contribution"]
        .sum()
        .reset_index(name="ece")
    )
    return bins_df.merge(ece, on=[col for col in GROUP_COLS if col in bins_df.columns], how="left")


def threshold_curve(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for base, sub in iter_groups(df, GROUP_COLS):
        sub = sub.dropna(subset=["confidence"])
        if sub.empty:
            continue
        labels = sorted(set(sub["true_label"].astype(str)) | set(sub["pred_label"].astype(str)))
        for threshold in thresholds:
            kept = sub.loc[sub["confidence"] >= threshold]
            row = dict(base)
            row["threshold"] = float(threshold)
            row["n_total"] = int(len(sub))
            row["n_kept"] = int(len(kept))
            row["coverage"] = float(len(kept) / len(sub)) if len(sub) else np.nan
            row["abstain_rate"] = float(1.0 - row["coverage"]) if len(sub) else np.nan
            if kept.empty:
                row["accuracy"] = np.nan
                row["macro_f1"] = np.nan
                row["mean_confidence"] = np.nan
            else:
                row["accuracy"] = float(accuracy_score(kept["true_label"].astype(str), kept["pred_label"].astype(str)))
                row["macro_f1"] = float(
                    f1_score(
                        kept["true_label"].astype(str),
                        kept["pred_label"].astype(str),
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                row["mean_confidence"] = float(kept["confidence"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_policy_summary(curve: pd.DataFrame, min_coverage: float = 0.5) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    group_cols = [col for col in GROUP_COLS if col in curve.columns]
    rows = []
    for base, sub in iter_groups(curve, group_cols):
        work = sub.copy()
        work["coverage_ok"] = pd.to_numeric(work["coverage"], errors="coerce").ge(float(min_coverage))
        candidates = work.loc[work["coverage_ok"]].copy()
        if candidates.empty:
            candidates = work.copy()
            selection_note = "best_macro_f1_without_min_coverage"
        else:
            selection_note = "best_macro_f1_with_min_coverage"
        candidates["macro_f1_sort"] = pd.to_numeric(candidates["macro_f1"], errors="coerce").fillna(-1.0)
        candidates["coverage_sort"] = pd.to_numeric(candidates["coverage"], errors="coerce").fillna(-1.0)
        selected = candidates.sort_values(["macro_f1_sort", "coverage_sort", "threshold"], ascending=[False, False, True]).iloc[0]
        rows.append({
            **base,
            "policy_min_coverage": float(min_coverage),
            "selected_threshold": float(selected["threshold"]),
            "selected_macro_f1": float(selected["macro_f1"]) if pd.notna(selected["macro_f1"]) else np.nan,
            "selected_accuracy": float(selected["accuracy"]) if pd.notna(selected["accuracy"]) else np.nan,
            "selected_coverage": float(selected["coverage"]) if pd.notna(selected["coverage"]) else np.nan,
            "selected_abstain_rate": float(selected["abstain_rate"]) if pd.notna(selected["abstain_rate"]) else np.nan,
            "selected_mean_confidence": float(selected["mean_confidence"]) if pd.notna(selected["mean_confidence"]) else np.nan,
            "selection_note": selection_note,
        })
    return pd.DataFrame(rows)


def build_calibration_paper_summary(summary: pd.DataFrame, bins: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_cols = [col for col in GROUP_COLS if col in summary.columns]
    out = summary.copy()
    if not bins.empty and "ece" in bins.columns:
        ece_cols = [col for col in GROUP_COLS if col in bins.columns]
        ece = bins.groupby(ece_cols, dropna=False)["ece"].max().reset_index()
        merge_cols = [col for col in group_cols if col in ece.columns]
        if merge_cols:
            out = out.merge(ece, on=merge_cols, how="left")
    if not policy.empty:
        policy_cols = [
            col for col in [
                *GROUP_COLS, "policy_min_coverage", "selected_threshold", "selected_macro_f1",
                "selected_accuracy", "selected_coverage", "selected_abstain_rate",
                "selected_mean_confidence", "selection_note",
            ] if col in policy.columns
        ]
        merge_cols = [col for col in group_cols if col in policy_cols]
        if merge_cols:
            out = out.merge(policy[policy_cols], on=merge_cols, how="left")
    gap = pd.to_numeric(out.get("confidence_accuracy_gap", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["calibration_direction"] = np.select(
        [gap.gt(0.05), gap.lt(-0.05)],
        ["over_confident", "under_confident"],
        default="roughly_calibrated",
    )
    leading = [
        *group_cols, "n", "accuracy", "mean_confidence", "confidence_accuracy_gap",
        "calibration_direction", "ece", "mean_brier_score", "selected_threshold",
        "selected_macro_f1", "selected_coverage", "selected_abstain_rate", "selection_note",
    ]
    cols = [col for col in leading if col in out.columns] + [col for col in out.columns if col not in leading]
    return out[cols]
def validate_prediction_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    if df.empty:
        raise FileNotFoundError(
            "No prediction rows found for calibration. "
            f"prediction_rows={args.prediction_rows or '<auto>'}, "
            f"results_dir={args.results_dir}, mode={args.mode}"
        )
    if "confidence" not in df.columns and "model_confidence" in df.columns:
        df = df.copy()
        df["confidence"] = df["model_confidence"]
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Prediction rows are missing required calibration columns: "
            + ", ".join(missing)
        )
    return df


def run(args):
    if args.prediction_rows:
        df = pd.read_csv(args.prediction_rows)
    else:
        df = load_prediction_rows(Path(args.results_dir), args.mode)
    df = validate_prediction_rows(df, args)
    df = add_calibration_columns(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = calibration_summary(df)
    bins = ece_bins(df, args.bins)
    curve = threshold_curve(df, parse_thresholds(args.thresholds))
    policy = calibration_policy_summary(curve, args.policy_min_coverage)
    paper_summary = build_calibration_paper_summary(summary, bins, policy)

    summary_path = out_dir / "calibration_summary.csv"
    bins_path = out_dir / "calibration_ece_bins.csv"
    curve_path = out_dir / "confidence_threshold_curve.csv"
    policy_path = out_dir / "calibration_policy_summary.csv"
    paper_summary_path = out_dir / "calibration_summary_paper.csv"
    manifest_path = out_dir / "calibration_manifest.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    curve.to_csv(curve_path, index=False, encoding="utf-8-sig")
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    paper_summary.to_csv(paper_summary_path, index=False, encoding="utf-8-sig")
    if getattr(args, "paper_tables_dir", ""):
        paper_tables_dir = Path(args.paper_tables_dir)
        paper_tables_dir.mkdir(parents=True, exist_ok=True)
        paper_name = args.paper_summary_name or ("calibration_summary_paper_locked_test.csv" if args.mode == "locked_test" else "calibration_summary_paper.csv")
        paper_summary.to_csv(paper_tables_dir / paper_name, index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": args.results_dir,
        "prediction_rows": args.prediction_rows,
        "mode": args.mode,
        "bins": args.bins,
        "thresholds": parse_thresholds(args.thresholds),
        "policy_min_coverage": float(args.policy_min_coverage),
        "n_prediction_rows": int(len(df)),
        "outputs": {
            "summary": str(summary_path),
            "ece_bins": str(bins_path),
            "threshold_curve": str(curve_path),
            "policy_summary": str(policy_path),
            "paper_summary": str(paper_summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, bins, curve, policy, paper_summary, manifest


def main():
    parser = argparse.ArgumentParser(description="Compute ECE, Brier score, and confidence threshold curves.")
    parser.add_argument("--results_dir", default="results_v74")
    parser.add_argument("--prediction_rows", default="")
    parser.add_argument("--output_dir", default="results_v74/calibration_analysis")
    parser.add_argument("--paper-tables-dir", default="", help="Optional paper_v74/tables directory for calibration_summary_paper*.csv.")
    parser.add_argument("--paper-summary-name", default="", help="Optional output filename under --paper-tables-dir.")
    parser.add_argument("--mode", default="no_support", choices=["configured", "no_support", "locked_test", "both", "all"])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--thresholds", default="0.0,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--policy-min-coverage", type=float, default=0.5)
    args = parser.parse_args()
    summary, _, _, _, _, manifest = run(args)
    print(summary.head(30).to_string(index=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
