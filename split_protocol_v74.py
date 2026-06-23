import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import ml_baselines_v74 as prep


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sample_key(df: pd.DataFrame) -> pd.Series:
    return df["case_id"].astype(str) + "__" + df["ear_side"].astype(str)


def class_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df[label_col].astype(str).value_counts().sort_index().items()}


def locked_test_score(full_df: pd.DataFrame, trainval_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str):
    train_missing = 0
    test_missing = 0
    distribution_gap = 0.0
    full_counts = full_df[label_col].astype(str).value_counts()
    train_counts = trainval_df[label_col].astype(str).value_counts()
    test_counts = test_df[label_col].astype(str).value_counts()
    for cls, total in full_counts.items():
        train_count = int(train_counts.get(cls, 0))
        test_count = int(test_counts.get(cls, 0))
        if train_count == 0:
            train_missing += 1
        if total >= 2 and test_count == 0:
            test_missing += 1
        full_ratio = float(total) / max(len(full_df), 1)
        test_ratio = float(test_count) / max(len(test_df), 1)
        distribution_gap += abs(full_ratio - test_ratio)
    return (train_missing, test_missing, distribution_gap)


def split_locked_test(df: pd.DataFrame, task_name: str, seed: int, locked_test_ratio: float):
    if locked_test_ratio <= 0:
        return df.copy(), df.iloc[:0].copy(), {
            "_locked_test_enabled": False,
            "_locked_test_ratio": 0.0,
        }

    label_col = prep.TASK_INFO[task_name]["label_cols"][0]
    if "case_id" not in df.columns or df["case_id"].nunique() < 2:
        raise ValueError(f"{task_name} cannot create locked test split without at least two case_id groups")

    candidates = []
    n_candidate_seeds = 200 if len(df) < 500 else 50
    groups = df["case_id"].astype(str)
    for offset in range(n_candidate_seeds):
        candidate_seed = seed * 1000 + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=locked_test_ratio, random_state=candidate_seed)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        trainval_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        score = locked_test_score(df, trainval_df, test_df, label_col)
        candidates.append((score, candidate_seed, trainval_df, test_df))

    best_score, selected_seed, trainval_df, test_df = min(candidates, key=lambda item: item[0])
    summary = {
        "_locked_test_enabled": True,
        "_locked_test_ratio": float(locked_test_ratio),
        "_locked_test_candidate_seed": int(selected_seed),
        "_locked_test_score_train_missing": int(best_score[0]),
        "_locked_test_score_test_missing": int(best_score[1]),
        "_locked_test_score_distribution_gap": float(best_score[2]),
    }
    return trainval_df.reset_index(drop=True), test_df.reset_index(drop=True), summary


def assignment_rows(df: pd.DataFrame, task_name: str, seed: int, split_name: str, label_col: str):
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "task": task_name,
            "seed": int(seed),
            "split": split_name,
            "case_id": row.get("case_id"),
            "ear_side": row.get("ear_side"),
            "sample_key": f"{row.get('case_id')}__{row.get('ear_side')}",
            "label_col": label_col,
            "label": row.get(label_col),
        })
    return rows


def split_summary_row(
    task_name: str,
    seed: int,
    split_name: str,
    df: pd.DataFrame,
    label_col: str,
    extra: Dict[str, object] = None,
):
    row = {
        "task": task_name,
        "seed": int(seed),
        "split": split_name,
        "n_rows": int(len(df)),
        "n_case_ids": int(df["case_id"].astype(str).nunique()) if "case_id" in df.columns else None,
        "label_col": label_col,
        "class_counts": json_dumps(class_counts(df, label_col)) if len(df) else "{}",
    }
    if extra:
        row.update(extra)
    return row


def build_protocol(data_dir: Path, seeds: List[int], locked_test_ratio: float):
    raw_dfs, feature_sets = prep.build_feature_sets(data_dir, "task")
    assignment = []
    summary = []

    for task_name, raw_df in raw_dfs.items():
        label_col = prep.TASK_INFO[task_name]["label_cols"][0]
        feature_cols = feature_sets[task_name]
        base_df = prep.pad_and_clean(raw_df, feature_cols)
        base_df, missing_stats = prep.clean_label_columns(base_df, [label_col])

        for seed in seeds:
            trainval_df, test_df, test_summary = split_locked_test(base_df, task_name, seed, locked_test_ratio)
            train_df, val_df, split_meta = prep.split_dataframe(trainval_df, task_name, seed)

            assignment.extend(assignment_rows(train_df, task_name, seed, "train", label_col))
            assignment.extend(assignment_rows(val_df, task_name, seed, "val", label_col))
            if len(test_df):
                assignment.extend(assignment_rows(test_df, task_name, seed, "test", label_col))

            train_cases = set(train_df["case_id"].astype(str))
            val_cases = set(val_df["case_id"].astype(str))
            test_cases = set(test_df["case_id"].astype(str)) if len(test_df) else set()
            common_extra = {
                "rows_before_label_cleaning": int(missing_stats["rows_before"]),
                "rows_after_label_cleaning": int(missing_stats["rows_after"]),
                "train_val_case_overlap": int(len(train_cases & val_cases)),
                "train_test_case_overlap": int(len(train_cases & test_cases)),
                "val_test_case_overlap": int(len(val_cases & test_cases)),
                "split_meta": json_dumps(split_meta),
                "locked_test_meta": json_dumps(test_summary),
            }
            summary.append(split_summary_row(task_name, seed, "train", train_df, label_col, common_extra))
            summary.append(split_summary_row(task_name, seed, "val", val_df, label_col, common_extra))
            if len(test_df):
                summary.append(split_summary_row(task_name, seed, "test", test_df, label_col, common_extra))

    return pd.DataFrame(assignment), pd.DataFrame(summary)


def save_protocol(assignment_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "split_manifest.csv"
    summary_path = output_dir / "split_summary.csv"
    manifest_path = output_dir / "split_protocol_manifest.json"
    assignment_df.to_csv(assignment_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "seeds": args.seeds,
        "locked_test_ratio": float(args.locked_test_ratio),
        "files": {
            "split_manifest": str(assignment_path),
            "split_summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_seeds(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Create repeated grouped split manifests for v7.4 tasks.")
    parser.add_argument("--data_dir", default=".", help="Directory containing task CSV files.")
    parser.add_argument("--output_dir", default="results_v74/split_protocol", help="Output directory.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated split seeds.")
    parser.add_argument("--locked_test_ratio", type=float, default=0.0, help="Optional grouped locked test ratio.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    assignment_df, summary_df = build_protocol(Path(args.data_dir), seeds, args.locked_test_ratio)
    manifest = save_protocol(assignment_df, summary_df, Path(args.output_dir), args)
    print(json.dumps({
        "status": "ok",
        "assignment_rows": int(len(assignment_df)),
        "summary_rows": int(len(summary_df)),
        **manifest,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
