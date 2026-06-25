import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpoint_utils_v74 import resolve_checkpoint_paths
from artificial_missingness_v74 import (
    DEFAULT_CHECKPOINT,
    load_model,
    load_task_data,
    metric_dict,
    normalize_features,
    parse_checkpoint_identity,
)
from preprocessing_v74 import TASK_INFO


FEATURE_GROUPS = {
    "Task1": {
        "ac_thresholds": ["ac_500Hz", "ac_1000Hz", "ac_2000Hz", "ac_4000Hz"],
        "ac_pta": ["ac_PTA"],
        "all_ac_evidence": ["ac_500Hz", "ac_1000Hz", "ac_2000Hz", "ac_4000Hz", "ac_PTA"],
    },
    "Task2": {
        "ac_thresholds": ["ac_500Hz", "ac_1000Hz", "ac_2000Hz", "ac_4000Hz", "ac_6000Hz", "ac_8000Hz"],
        "ac_core_thresholds": ["ac_500Hz", "ac_1000Hz", "ac_2000Hz", "ac_4000Hz"],
        "ac_high_frequency_thresholds": ["ac_6000Hz", "ac_8000Hz"],
        "ac_nr_flags": ["ac_500Hz_nr", "ac_1000Hz_nr", "ac_2000Hz_nr", "ac_4000Hz_nr", "ac_6000Hz_nr", "ac_8000Hz_nr"],
        "ac_high_frequency_nr_flags": ["ac_6000Hz_nr", "ac_8000Hz_nr"],
        "bc_thresholds": ["bc_500Hz", "bc_1000Hz", "bc_2000Hz", "bc_4000Hz"],
        "bc_nr_flags": ["bc_500Hz_nr", "bc_1000Hz_nr", "bc_2000Hz_nr", "bc_4000Hz_nr"],
        "bc_missing_flags": ["bc_500Hz_missing", "bc_1000Hz_missing", "bc_2000Hz_missing", "bc_4000Hz_missing"],
        "abg_values": ["abg_500Hz", "abg_1000Hz", "abg_2000Hz", "abg_4000Hz"],
        "abg_missing_flags": ["abg_500Hz_missing", "abg_1000Hz_missing", "abg_2000Hz_missing", "abg_4000Hz_missing"],
        "abg_censored_flags": ["abg_500Hz_censored", "abg_1000Hz_censored", "abg_2000Hz_censored", "abg_4000Hz_censored"],
        "all_bc_evidence": [
            "bc_500Hz", "bc_1000Hz", "bc_2000Hz", "bc_4000Hz",
            "bc_500Hz_nr", "bc_1000Hz_nr", "bc_2000Hz_nr", "bc_4000Hz_nr",
            "bc_500Hz_missing", "bc_1000Hz_missing", "bc_2000Hz_missing", "bc_4000Hz_missing",
        ],
        "all_ac_evidence": [
            "ac_500Hz", "ac_1000Hz", "ac_2000Hz", "ac_4000Hz", "ac_6000Hz", "ac_8000Hz",
            "ac_500Hz_nr", "ac_1000Hz_nr", "ac_2000Hz_nr", "ac_4000Hz_nr", "ac_6000Hz_nr", "ac_8000Hz_nr",
        ],
        "all_abg_evidence": [
            "abg_500Hz", "abg_1000Hz", "abg_2000Hz", "abg_4000Hz",
            "abg_500Hz_missing", "abg_1000Hz_missing", "abg_2000Hz_missing", "abg_4000Hz_missing",
            "abg_500Hz_censored", "abg_1000Hz_censored", "abg_2000Hz_censored", "abg_4000Hz_censored",
        ],
    },
    "Task3": {
        "tymp_numeric_values": ["tymp_Vea", "tymp_peak_daPa", "tymp_peak_mmho", "tymp_Width_daPa"],
        "tymp_peak_evidence": ["tymp_peak_daPa", "tymp_peak_mmho"],
        "tymp_width_evidence": ["tymp_Width_daPa"],
        "tymp_real_zero_flags": [
            "tymp_Vea_real_zero", "tymp_peak_daPa_real_zero", "tymp_peak_mmho_real_zero", "tymp_Width_daPa_real_zero",
        ],
        "tymp_missing_zero_flags": [
            "tymp_Vea_missing_zero", "tymp_peak_daPa_missing_zero", "tymp_peak_mmho_missing_zero", "tymp_Width_daPa_missing_zero",
        ],
        "tymp_np_zero_flags": [
            "tymp_Vea_np_zero", "tymp_peak_daPa_np_zero", "tymp_peak_mmho_np_zero", "tymp_Width_daPa_np_zero",
        ],
        "all_tympanogram_evidence": [
            "tymp_Vea", "tymp_peak_daPa", "tymp_peak_mmho", "tymp_Width_daPa",
            "tymp_Vea_real_zero", "tymp_peak_daPa_real_zero", "tymp_peak_mmho_real_zero", "tymp_Width_daPa_real_zero",
            "tymp_Vea_missing_zero", "tymp_peak_daPa_missing_zero", "tymp_peak_mmho_missing_zero", "tymp_Width_daPa_missing_zero",
            "tymp_Vea_np_zero", "tymp_peak_daPa_np_zero", "tymp_peak_mmho_np_zero", "tymp_Width_daPa_np_zero",
        ],
    },
}

REFLEX_NOTE = "No Acoustic Reflex feature columns are present in the current v7.4 CSV inputs."


def resolve_checkpoints(args) -> list[str]:
    paths = resolve_checkpoint_paths(
        checkpoint=args.checkpoint,
        checkpoints=args.checkpoints,
        checkpoint_glob=args.checkpoint_glob,
        default_checkpoint=DEFAULT_CHECKPOINT,
        search_defaults=True,
    )
    return [str(path) for path in paths]


def available_group_indices(task_name: str, union_features: list[str]) -> list[dict]:
    feature_to_idx = {feature: idx for idx, feature in enumerate(union_features)}
    rows = []
    for group_name, features in FEATURE_GROUPS.get(task_name, {}).items():
        present = [feature for feature in features if feature in feature_to_idx]
        if not present:
            continue
        rows.append({
            "feature_group": group_name,
            "features": present,
            "indices": [feature_to_idx[feature] for feature in present],
        })
    return rows


def predict_from_matrix(model, meta: dict, X: np.ndarray, task_name: str, batch_size: int, device: torch.device):
    label_col = TASK_INFO[task_name]["label_cols"][0]
    class_names = meta["tasks"][task_name]["class_names"][label_col]
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


def metric_row(base_info: dict, task_name: str, analysis_type: str, feature_group: str, features: list[str], y_true, y_pred, base_metrics: dict, repeat=None) -> dict:
    metrics = metric_dict(task_name, y_true, y_pred)
    row = {
        **base_info,
        "task": task_name,
        "analysis_type": analysis_type,
        "feature_group": feature_group,
        "feature_count": int(len(features)),
        "features": ",".join(features),
        "repeat": repeat,
        "reflex_note": REFLEX_NOTE,
    }
    row.update(metrics)
    for metric in ["accuracy", "macro_f1", "balanced_accuracy", "macro_sensitivity", "macro_specificity"]:
        row[f"base_{metric}"] = base_metrics.get(metric)
        row[f"{metric}_drop_from_base"] = base_metrics.get(metric) - metrics.get(metric)
    return row


def summarize_permutation(permutation_rows: list[dict]) -> pd.DataFrame:
    if not permutation_rows:
        return pd.DataFrame()
    df = pd.DataFrame(permutation_rows)
    group_cols = [
        "checkpoint", "config_name", "exp_name", "seed", "task", "analysis_type",
        "feature_group", "feature_count", "features", "reflex_note",
    ]
    group_cols = [col for col in group_cols if col in df.columns]
    value_cols = [
        "accuracy", "macro_f1", "balanced_accuracy", "macro_sensitivity", "macro_specificity",
        "accuracy_drop_from_base", "macro_f1_drop_from_base", "balanced_accuracy_drop_from_base",
        "macro_sensitivity_drop_from_base", "macro_specificity_drop_from_base",
    ]
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row["n_repeats"] = int(len(sub))
        for col in value_cols:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                row[f"{col}__mean"] = float(values.mean()) if values.notna().any() else np.nan
                row[f"{col}__std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values([col for col in ["task", "macro_f1_drop_from_base__mean", "feature_group"] if col in rows[0] if rows]).reset_index(drop=True)


def run_feature_importance(args):
    device = torch.device(args.device)
    rng = np.random.default_rng(args.random_state)
    checkpoints = resolve_checkpoints(args)
    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    ablation_rows = []
    permutation_rows = []
    baseline_rows = []

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
            df = load_task_data(Path(args.data_dir), task_name)
            label_col = TASK_INFO[task_name]["label_cols"][0]
            y_true = df[label_col].astype(str).tolist()
            X = normalize_features(df, task_name, meta)
            y_base, base_confidences = predict_from_matrix(model, meta, X, task_name, args.batch_size, device)
            base_metrics = metric_dict(task_name, y_true, y_base)
            baseline_rows.append({
                **base_info,
                "task": task_name,
                "analysis_type": "baseline",
                "feature_group": "none",
                "feature_count": 0,
                "features": "",
                "mean_confidence": float(np.mean(base_confidences)) if base_confidences else np.nan,
                "reflex_note": REFLEX_NOTE,
                **base_metrics,
            })

            for group in available_group_indices(task_name, meta["union_features"]):
                features = group["features"]
                indices = group["indices"]

                X_ablate = X.copy()
                X_ablate[:, indices] = 0.0
                y_ablate, _ = predict_from_matrix(model, meta, X_ablate, task_name, args.batch_size, device)
                ablation_rows.append(
                    metric_row(base_info, task_name, "mean_mask_ablation", group["feature_group"], features, y_true, y_ablate, base_metrics)
                )

                if len(X) >= 2:
                    for repeat in range(args.permutation_repeats):
                        X_perm = X.copy()
                        for idx in indices:
                            X_perm[:, idx] = rng.permutation(X_perm[:, idx])
                        y_perm, _ = predict_from_matrix(model, meta, X_perm, task_name, args.batch_size, device)
                        permutation_rows.append(
                            metric_row(base_info, task_name, "permutation", group["feature_group"], features, y_true, y_perm, base_metrics, repeat=repeat)
                        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_df = pd.DataFrame(baseline_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    permutation_detail_df = pd.DataFrame(permutation_rows)
    permutation_summary_df = summarize_permutation(permutation_rows)

    baseline_df.to_csv(out_dir / "feature_importance_baseline.csv", index=False, encoding="utf-8-sig")
    ablation_df.to_csv(out_dir / "feature_group_ablation_summary.csv", index=False, encoding="utf-8-sig")
    permutation_detail_df.to_csv(out_dir / "feature_group_permutation_detail.csv", index=False, encoding="utf-8-sig")
    permutation_summary_df.to_csv(out_dir / "feature_group_permutation_importance.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "checkpoints": checkpoints,
        "tasks": requested_tasks,
        "batch_size": args.batch_size,
        "permutation_repeats": args.permutation_repeats,
        "random_state": args.random_state,
        "feature_groups": FEATURE_GROUPS,
        "reflex_note": REFLEX_NOTE,
        "ablation_interpretation": "Inference-time mean masking; this is not retraining ablation.",
        "permutation_interpretation": "Group-level permutation importance on normalized model inputs.",
        "outputs": {
            "baseline": str(out_dir / "feature_importance_baseline.csv"),
            "ablation_summary": str(out_dir / "feature_group_ablation_summary.csv"),
            "permutation_detail": str(out_dir / "feature_group_permutation_detail.csv"),
            "permutation_importance": str(out_dir / "feature_group_permutation_importance.csv"),
        },
    }
    (out_dir / "feature_importance_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return baseline_df, ablation_df, permutation_summary_df


def main():
    parser = argparse.ArgumentParser(description="Feature-group importance and inference-time ablation for v7.4 models.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint paths.")
    parser.add_argument("--checkpoint-glob", default=None, help="Glob such as results_v74/five_runs/**/best_model.pth.")
    parser.add_argument("--output-dir", default="results_v74/feature_importance")
    parser.add_argument("--tasks", default="Task1,Task2,Task3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    baseline, ablation, permutation = run_feature_importance(args)
    print("baseline_rows=", len(baseline))
    print("ablation_rows=", len(ablation))
    print("permutation_groups=", len(permutation))


if __name__ == "__main__":
    main()
