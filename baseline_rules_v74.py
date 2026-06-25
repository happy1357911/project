import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from clinical_rules_v74 import (
    MISSING_TOKENS,
    TASK2_ABG_FREQS,
    TASK2_AC_NR_LIMITS_DB,
    TASK2_BC_NR_LIMITS_DB,
    TASK2_STANDARD_HEARING_TYPES,
    TASK3_NP_EXTREME_VALUES,
    infer_task2_hearing_type_from_rule,
    nr_indicator_series,
    numeric_series,
    predict_task3_type_from_rule,
    sanitize_column_name,
    clean_value_none as clean_value,
)
from preprocessing_v74 import (
    evidence_status as shared_evidence_status,
    normalize_prediction,
    prepare_task1_dataframe as shared_prepare_task1_dataframe,
    prepare_task2_dataframe as shared_prepare_task2_dataframe,
    load_tabular_data,
    prepare_task3_dataframe as shared_prepare_task3_dataframe,
    rule_prediction as shared_rule_prediction,
)


TASK1_CSV = "task1_all_three_common14_v1.csv"
TASK2_CSV = "task2_3_pure_data(6_24).xlsx"
TASK3_CSV = "task2_3_pure_data(6_24).xlsx"
INSUFFICIENT_EVIDENCE_LABEL = "INSUFFICIENT_EVIDENCE"


def col_or_nan(df, col):
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def load_csv(data_dir, filename):
    path = Path(data_dir) / filename
    return load_tabular_data(path)


def build_task1_ears(df):
    df = df.copy()
    df.columns = [sanitize_column_name(col) for col in df.columns]
    frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = np.arange(len(df))
        ear_df["ear_side"] = side
        ear_df["label"] = col_or_nan(df, f"hearing_degree_WHO_PTAbased_{side}")
        ear_df["ac_PTA"] = numeric_series(df, f"{side}_PTA")
        frames.append(ear_df)
    out = pd.concat(frames, ignore_index=True)
    out["label"] = out["label"].astype(str).str.strip()
    return out.dropna(subset=["label"]).reset_index(drop=True)


def predict_task1_degree(pta):
    pta = clean_value(pta)
    if pta is None:
        return None
    if pta < 20:
        return "Normal hearing"
    if pta < 35:
        return "Mild hearing loss"
    if pta < 50:
        return "Moderate hearing loss"
    if pta < 65:
        return "Moderately severe hearing loss"
    if pta < 80:
        return "Severe hearing loss"
    if pta < 95:
        return "Profound hearing loss"
    return "Complete or total hearing loss"


def build_task2_ears(df):
    df = df.copy()
    df.columns = [sanitize_column_name(col) for col in df.columns]
    frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = np.arange(len(df))
        ear_df["ear_side"] = side
        ear_df["label"] = col_or_nan(df, f"hearing_type_{side}").astype(str).str.strip().str.upper()
        for hz in TASK2_ABG_FREQS:
            ac_col = f"{side}_{hz}Hz"
            bc_col = f"bc_{side}_{hz}Hz"
            ear_df[f"ac_{hz}Hz"] = numeric_series(df, ac_col, TASK2_AC_NR_LIMITS_DB)
            ear_df[f"ac_{hz}Hz_nr"] = nr_indicator_series(df, ac_col)
            ear_df[f"bc_{hz}Hz"] = numeric_series(df, bc_col, TASK2_BC_NR_LIMITS_DB)
            ear_df[f"bc_{hz}Hz_nr"] = nr_indicator_series(df, bc_col)
            ear_df[f"bc_{hz}Hz_missing"] = ear_df[f"bc_{hz}Hz"].isna().astype(float)
            ear_df[f"abg_{hz}Hz"] = ear_df[f"ac_{hz}Hz"] - ear_df[f"bc_{hz}Hz"]
            ear_df[f"abg_{hz}Hz_missing"] = (
                ear_df[f"ac_{hz}Hz"].isna() | ear_df[f"bc_{hz}Hz"].isna()
            ).astype(float)
            ear_df[f"abg_{hz}Hz_censored"] = (
                (ear_df[f"ac_{hz}Hz_nr"] >= 0.5)
                | (ear_df[f"bc_{hz}Hz_nr"] >= 0.5)
            ).astype(float)
        frames.append(ear_df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["label"].isin(TASK2_STANDARD_HEARING_TYPES)].reset_index(drop=True)
    return out


def predict_task2_type(row):
    return infer_task2_hearing_type_from_rule(row)[0]


def build_task3_ears(df):
    df = df.copy()
    df.columns = [sanitize_column_name(col) for col in df.columns]
    frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = np.arange(len(df))
        ear_df["ear_side"] = side
        ear_df["label"] = col_or_nan(df, f"tymp_{side}_type").astype(str).str.strip().str.upper()
        for src_suffix, neutral_col in [
            ("Vea", "tymp_Vea"),
            ("peak_daPa", "tymp_peak_daPa"),
            ("peak_mmho", "tymp_peak_mmho"),
            ("Width_daPa", "tymp_Width_daPa"),
        ]:
            src_col = f"tymp_{side}_{src_suffix}"
            raw = col_or_nan(df, src_col).astype(str).str.strip()
            numeric = pd.to_numeric(col_or_nan(df, src_col), errors="coerce")
            is_np = raw.str.upper().eq("NP")
            is_missing = raw.isin(MISSING_TOKENS) | raw.isna()
            ear_df[neutral_col] = numeric
            if neutral_col in TASK3_NP_EXTREME_VALUES:
                ear_df.loc[is_np, neutral_col] = TASK3_NP_EXTREME_VALUES[neutral_col]
            ear_df[f"{neutral_col}_np_zero"] = is_np.astype(float)
            ear_df[f"{neutral_col}_missing_zero"] = is_missing.astype(float)
        frames.append(ear_df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["label"].isin({"A", "B", "C"})].reset_index(drop=True)
    return out


def predict_task3_type(row):
    return predict_task3_type_from_rule(row)


def task1_evidence_status(row):
    has_pta = clean_value(row.get("ac_PTA")) is not None
    return {
        "evidence_status": "complete_evidence" if has_pta else "no_pta_data",
        "baseline_covered": bool(has_pta),
        "has_missing": bool(not has_pta),
    }


def task2_evidence_status(row):
    has_ac_missing = False
    has_bc_missing = False
    has_any_nr = False
    has_abg_censored = False
    has_abg_missing = False
    bc_present_count = 0
    ac_present_count = 0

    for hz in TASK2_ABG_FREQS:
        ac_col = f"ac_{hz}Hz"
        bc_col = f"bc_{hz}Hz"
        abg_col = f"abg_{hz}Hz"
        ac_value = clean_value(row.get(ac_col))
        bc_value = clean_value(row.get(bc_col))
        ac_nr = clean_value(row.get(f"{ac_col}_nr")) or 0.0
        bc_nr = clean_value(row.get(f"{bc_col}_nr")) or 0.0
        bc_missing_flag = clean_value(row.get(f"{bc_col}_missing")) or 0.0
        abg_missing_flag = clean_value(row.get(f"{abg_col}_missing")) or 0.0
        abg_censored_flag = clean_value(row.get(f"{abg_col}_censored")) or 0.0

        ac_missing = ac_value is None and ac_nr < 0.5
        bc_missing = (bc_value is None and bc_nr < 0.5) or bc_missing_flag >= 0.5
        abg_missing = abg_missing_flag >= 0.5 or ac_missing or bc_missing
        abg_censored = abg_censored_flag >= 0.5 or ac_nr >= 0.5 or bc_nr >= 0.5

        has_ac_missing = has_ac_missing or ac_missing
        has_bc_missing = has_bc_missing or bc_missing
        has_any_nr = has_any_nr or ac_nr >= 0.5 or bc_nr >= 0.5
        has_abg_missing = has_abg_missing or abg_missing
        has_abg_censored = has_abg_censored or abg_censored
        ac_present_count += int(ac_value is not None)
        bc_present_count += int(bc_value is not None)

    no_bc_data = bc_present_count == 0
    if no_bc_data:
        status = "no_bc_data"
    elif has_any_nr or has_abg_censored:
        status = "nr_or_censored"
    elif has_bc_missing:
        status = "bc_missing"
    elif has_ac_missing:
        status = "ac_missing"
    elif has_abg_missing:
        status = "abg_missing"
    else:
        status = "complete_evidence"

    covered = status == "complete_evidence"
    return {
        "evidence_status": status,
        "baseline_covered": bool(covered),
        "has_ac_missing": bool(has_ac_missing),
        "has_bc_missing": bool(has_bc_missing),
        "has_any_nr": bool(has_any_nr),
        "has_abg_missing": bool(has_abg_missing),
        "has_abg_censored": bool(has_abg_censored),
        "no_bc_data": bool(no_bc_data),
        "ac_present_count": int(ac_present_count),
        "bc_present_count": int(bc_present_count),
    }


def task3_evidence_status(row):
    np_cols = [col for col in row if str(col).endswith("_np_zero")]
    missing_cols = [col for col in row if str(col).endswith("_missing_zero")]
    has_np = any((clean_value(row.get(col)) or 0.0) >= 0.5 for col in np_cols)
    has_missing = any((clean_value(row.get(col)) or 0.0) >= 0.5 for col in missing_cols)
    if has_missing:
        status = "missing_tymp_data"
    elif has_np:
        status = "np_evidence"
    else:
        status = "complete_evidence"
    return {
        "evidence_status": status,
        "baseline_covered": bool(not has_missing),
        "has_np": bool(has_np),
        "has_missing": bool(has_missing),
    }


def build_prediction_rows(task_name, df, predictor, evidence_fn):
    rows = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        forced_pred = normalize_prediction(predictor(raw))
        evidence = evidence_fn(raw)
        baseline_covered = bool(evidence.get("baseline_covered", False))
        abstain_pred = forced_pred if baseline_covered else INSUFFICIENT_EVIDENCE_LABEL
        out = {
            "task": task_name,
            "case_id": raw.get("case_id"),
            "ear_side": raw.get("ear_side"),
            "true_label": str(raw.get("label")),
            "forced_pred_label": forced_pred,
            "rule_decision_label": forced_pred,
            "abstain_pred_label": abstain_pred,
            "baseline_covered": baseline_covered,
            "evidence_status": evidence.get("evidence_status", "unknown"),
            "forced_correct": forced_pred == str(raw.get("label")),
            "rule_decision_correct": forced_pred == str(raw.get("label")),
            "abstain_correct": abstain_pred == str(raw.get("label")),
        }
        out.update(evidence)
        rows.append(out)
    return pd.DataFrame(rows)


def evaluate_predictions(name, y_true, y_pred):
    y_true = [str(value) for value in y_true]
    y_pred = [normalize_prediction(value) for value in y_pred]
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "task": name,
        "n": int(len(y_true)),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(
            _balanced_accuracy_no_warning(y_true, y_pred)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _balanced_accuracy_no_warning(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return balanced_accuracy_score(y_true, y_pred)


def evaluate_baseline_task(task_name, pred_df):
    y_true = pred_df["true_label"].astype(str).tolist()
    forced_pred = pred_df["forced_pred_label"].astype(str).tolist()
    abstain_pred = pred_df["abstain_pred_label"].astype(str).tolist()
    covered_mask = pred_df["baseline_covered"].astype(bool)

    forced = evaluate_predictions(task_name, y_true, forced_pred)
    abstain_as_error = evaluate_predictions(task_name, y_true, abstain_pred)

    if covered_mask.any():
        covered = evaluate_predictions(
            task_name,
            pred_df.loc[covered_mask, "true_label"].astype(str).tolist(),
            pred_df.loc[covered_mask, "forced_pred_label"].astype(str).tolist(),
        )
    else:
        covered = {
            "task": task_name,
            "n": 0,
            "labels": sorted(set(y_true)),
            "accuracy": None,
            "macro_f1": None,
            "balanced_accuracy": None,
            "confusion_matrix": [],
        }

    n_total = int(len(pred_df))
    n_covered = int(covered_mask.sum())
    n_abstained = int(n_total - n_covered)
    evidence_counts = {
        str(k): int(v)
        for k, v in pred_df["evidence_status"].value_counts().sort_index().items()
    }
    return {
        "task": task_name,
        "n_total": n_total,
        "n_covered": n_covered,
        "n_abstained": n_abstained,
        "coverage": float(n_covered / n_total) if n_total else None,
        "abstain_rate": float(n_abstained / n_total) if n_total else None,
        "labels": forced["labels"],
        "evidence_counts": evidence_counts,
        "forced": forced,
        "covered_only": covered,
        "abstain_as_error": abstain_as_error,
    }


def build_evidence_summary(predictions):
    rows = []
    for (task, status), sub in predictions.groupby(["task", "evidence_status"], dropna=False):
        rows.append({
            "task": task,
            "evidence_status": status,
            "n": int(len(sub)),
            "covered": bool(sub["baseline_covered"].iloc[0]) if len(sub) else False,
            "forced_accuracy": float(sub["forced_correct"].mean()) if len(sub) else None,
            "abstain_accuracy": float(sub["abstain_correct"].mean()) if len(sub) else None,
        })
    return pd.DataFrame(rows).sort_values(["task", "evidence_status"]).reset_index(drop=True)


def evaluate_rule_baselines(data_dir):
    task1 = shared_prepare_task1_dataframe(load_csv(data_dir, TASK1_CSV)).rename(
        columns={"hearing_degree_WHO_PTAbased": "label"}
    )
    task1 = task1.dropna(subset=["label"]).reset_index(drop=True)
    task1_pred_rows = build_prediction_rows(
        "Task1",
        task1,
        lambda row: shared_rule_prediction("Task1", row),
        lambda row: shared_evidence_status("Task1", row),
    )
    task1_eval = evaluate_baseline_task("Task1", task1_pred_rows)

    task2 = shared_prepare_task2_dataframe(load_csv(data_dir, TASK2_CSV)).rename(
        columns={"hearing_type": "label"}
    )
    task2_pred_rows = build_prediction_rows(
        "Task2",
        task2,
        lambda row: shared_rule_prediction("Task2", row),
        lambda row: shared_evidence_status("Task2", row),
    )
    task2_eval = evaluate_baseline_task("Task2", task2_pred_rows)

    task3 = shared_prepare_task3_dataframe(load_csv(data_dir, TASK3_CSV)).rename(
        columns={"tymp_type": "label"}
    )
    task3 = task3[task3["label"].astype(str).str.strip().str.upper().isin({"A", "B", "C"})].reset_index(drop=True)
    task3_pred_rows = build_prediction_rows(
        "Task3",
        task3,
        lambda row: shared_rule_prediction("Task3", row),
        lambda row: shared_evidence_status("Task3", row),
    )
    task3_eval = evaluate_baseline_task("Task3", task3_pred_rows)

    predictions = pd.concat(
        [task1_pred_rows, task2_pred_rows, task3_pred_rows],
        ignore_index=True,
        sort=False,
    )

    return {
        "metrics": {
            "Task1": task1_eval,
            "Task2": task2_eval,
            "Task3": task3_eval,
        },
        "predictions": predictions,
        "evidence_summary": build_evidence_summary(predictions),
    }


def save_outputs(results, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rule_baseline_summary.json"
    csv_path = out_dir / "rule_baseline_summary.csv"
    pred_path = out_dir / "rule_baseline_predictions.csv"
    evidence_path = out_dir / "rule_baseline_evidence_summary.csv"
    metrics = results["metrics"]
    predictions = results["predictions"]
    evidence_summary = results["evidence_summary"]
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for task_name, task_metrics in metrics.items():
        rows.append({
            "task": task_name,
            "n_total": task_metrics["n_total"],
            "n_covered": task_metrics["n_covered"],
            "n_abstained": task_metrics["n_abstained"],
            "coverage": task_metrics["coverage"],
            "abstain_rate": task_metrics["abstain_rate"],
            "rule_decision_accuracy": task_metrics["forced"]["accuracy"],
            "rule_decision_macro_f1": task_metrics["forced"]["macro_f1"],
            "rule_decision_balanced_accuracy": task_metrics["forced"]["balanced_accuracy"],
            "forced_accuracy": task_metrics["forced"]["accuracy"],
            "forced_macro_f1": task_metrics["forced"]["macro_f1"],
            "forced_balanced_accuracy": task_metrics["forced"]["balanced_accuracy"],
            "covered_accuracy": task_metrics["covered_only"]["accuracy"],
            "covered_macro_f1": task_metrics["covered_only"]["macro_f1"],
            "covered_balanced_accuracy": task_metrics["covered_only"]["balanced_accuracy"],
            "abstain_as_error_accuracy": task_metrics["abstain_as_error"]["accuracy"],
            "abstain_as_error_macro_f1": task_metrics["abstain_as_error"]["macro_f1"],
            "abstain_as_error_balanced_accuracy": task_metrics["abstain_as_error"]["balanced_accuracy"],
            "labels": ",".join(task_metrics["labels"]),
            "evidence_counts": json.dumps(task_metrics["evidence_counts"], ensure_ascii=False, sort_keys=True),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    evidence_summary.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path, pred_path, evidence_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate v7.4 clinical rule baselines.")
    parser.add_argument("--data-dir", default=".", help="Directory containing the three task CSV files.")
    parser.add_argument("--output-dir", default="results_v74/rule_baselines", help="Directory for baseline metrics.")
    args = parser.parse_args()

    results = evaluate_rule_baselines(args.data_dir)
    json_path, csv_path, pred_path, evidence_path = save_outputs(results, args.output_dir)
    for task_name, metrics in results["metrics"].items():
        print(
            f"{task_name}: n={metrics['n_total']} "
            f"coverage={metrics['coverage']:.4f} "
            f"forced_accuracy={metrics['forced']['accuracy']:.4f} "
            f"forced_macro_f1={metrics['forced']['macro_f1']:.4f} "
            f"covered_macro_f1={metrics['covered_only']['macro_f1'] if metrics['covered_only']['macro_f1'] is not None else 'NA'} "
            f"abstain_rate={metrics['abstain_rate']:.4f}"
        )
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_predictions={pred_path}")
    print(f"saved_evidence_summary={evidence_path}")


if __name__ == "__main__":
    main()
