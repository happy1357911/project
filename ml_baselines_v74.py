import argparse
import importlib.util
import inspect
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from clinical_rules_v74 import (
    MISSING_TOKENS,
    TASK2_ABG_FREQS,
    TASK2_AC_NR_LIMITS_DB,
    TASK2_BC_NR_LIMITS_DB,
    TASK2_EAR_LEVEL_AC_FREQS,
    TASK2_STANDARD_HEARING_TYPES,
    TASK3_SIDE_NP_EXTREME_VALUES,
    nr_indicator_series,
    numeric_series,
)
from preprocessing_v74 import (
    load_tabular_data,
    prepare_task1_dataframe as shared_prepare_task1_dataframe,
    prepare_task2_dataframe as shared_prepare_task2_dataframe,
    prepare_task3_dataframe as shared_prepare_task3_dataframe,
)


LOW_SUPPORT_THRESHOLD = 5

TASK1_EAR_LEVEL_FEATURES = [
    "ac_500Hz",
    "ac_1000Hz",
    "ac_2000Hz",
    "ac_4000Hz",
    "ac_PTA",
]

TASK2_EAR_LEVEL_FEATURES = [
    "ac_500Hz",
    "ac_1000Hz",
    "ac_2000Hz",
    "ac_4000Hz",
    "ac_6000Hz",
    "ac_8000Hz",
    "ac_500Hz_nr",
    "ac_1000Hz_nr",
    "ac_2000Hz_nr",
    "ac_4000Hz_nr",
    "ac_6000Hz_nr",
    "ac_8000Hz_nr",
    "bc_500Hz",
    "bc_1000Hz",
    "bc_2000Hz",
    "bc_4000Hz",
    "bc_500Hz_nr",
    "bc_1000Hz_nr",
    "bc_2000Hz_nr",
    "bc_4000Hz_nr",
    "bc_500Hz_missing",
    "bc_1000Hz_missing",
    "bc_2000Hz_missing",
    "bc_4000Hz_missing",
    "abg_500Hz",
    "abg_1000Hz",
    "abg_2000Hz",
    "abg_4000Hz",
    "abg_500Hz_missing",
    "abg_1000Hz_missing",
    "abg_2000Hz_missing",
    "abg_4000Hz_missing",
    "abg_500Hz_censored",
    "abg_1000Hz_censored",
    "abg_2000Hz_censored",
    "abg_4000Hz_censored",
]

TASK3_BASE_FEATURES = [
    "tymp_right_Vea",
    "tymp_right_peak_daPa",
    "tymp_right_peak_mmho",
    "tymp_right_Width_daPa",
    "tymp_left_Vea",
    "tymp_left_peak_daPa",
    "tymp_left_peak_mmho",
    "tymp_left_Width_daPa",
]

TASK3_EAR_LEVEL_BASE_FEATURES = [
    "tymp_Vea",
    "tymp_peak_daPa",
    "tymp_peak_mmho",
    "tymp_Width_daPa",
]

TASK3_EAR_LEVEL_DERIVED_FEATURES = []
for _col in TASK3_EAR_LEVEL_BASE_FEATURES:
    TASK3_EAR_LEVEL_DERIVED_FEATURES.extend([
        f"{_col}_real_zero",
        f"{_col}_missing_zero",
        f"{_col}_np_zero",
    ])

TASK3_EAR_LEVEL_FEATURES = [
    *TASK3_EAR_LEVEL_BASE_FEATURES,
    *TASK3_EAR_LEVEL_DERIVED_FEATURES,
]

TASK_INFO = {
    "Task1": {
        "csv": "task1_all_three_common14_v1.csv",
        "label_cols": ["hearing_degree_WHO_PTAbased"],
        "feature_cols": TASK1_EAR_LEVEL_FEATURES,
    },
    "Task2": {
        "csv": "task2_3_pure_data(6_24).xlsx",
        "label_cols": ["hearing_type"],
        "feature_cols": TASK2_EAR_LEVEL_FEATURES,
    },
    "Task3": {
        "csv": "task2_3_pure_data(6_24).xlsx",
        "label_cols": ["tymp_type"],
        "feature_cols": TASK3_EAR_LEVEL_FEATURES,
    },
}

OPTIONAL_MODEL_PACKAGES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def sanitize_column_name(x: str) -> str:
    x = str(x).strip()
    return re.sub(r"\s+", "_", x)


def col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = np.arange(len(df))
    return df


def convert_task2_nr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for side in ["right", "left"]:
        for hz in TASK2_AC_NR_LIMITS_DB:
            ac_col = f"{side}_{hz}Hz"
            if ac_col in df.columns:
                df[f"{ac_col}_nr"] = nr_indicator_series(df, ac_col)
                df[ac_col] = numeric_series(df, ac_col, TASK2_AC_NR_LIMITS_DB)
        for hz in TASK2_ABG_FREQS:
            bc_col = f"bc_{side}_{hz}Hz"
            if bc_col in df.columns:
                df[f"{bc_col}_nr"] = nr_indicator_series(df, bc_col)
                df[bc_col] = numeric_series(df, bc_col, TASK2_BC_NR_LIMITS_DB)
    return df


def add_task2_measurement_features(df: pd.DataFrame) -> pd.DataFrame:
    derived_cols = {}
    for side in ["right", "left"]:
        for hz in TASK2_ABG_FREQS:
            ac_col = f"{side}_{hz}Hz"
            bc_col = f"bc_{side}_{hz}Hz"
            ac = numeric_series(df, ac_col, TASK2_AC_NR_LIMITS_DB)
            bc = numeric_series(df, bc_col, TASK2_BC_NR_LIMITS_DB)
            ac_nr = pd.to_numeric(df.get(f"{ac_col}_nr", 0.0), errors="coerce").fillna(0.0)
            bc_nr = pd.to_numeric(df.get(f"{bc_col}_nr", 0.0), errors="coerce").fillna(0.0)

            derived_cols[f"{bc_col}_missing"] = bc.isna().astype(float)
            derived_cols[f"abg_{side}_{hz}Hz"] = ac - bc
            derived_cols[f"abg_{side}_{hz}Hz_missing"] = (ac.isna() | bc.isna()).astype(float)
            abg_censored = ((ac_nr >= 0.5) | (bc_nr >= 0.5)) & ~(ac.isna() | bc.isna())
            derived_cols[f"abg_{side}_{hz}Hz_censored"] = abg_censored.astype(float)
    return pd.concat([df, pd.DataFrame(derived_cols, index=df.index)], axis=1)


def build_task1_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ear_frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["hearing_degree_WHO_PTAbased"] = col_or_nan(df, f"hearing_degree_WHO_PTAbased_{side}")
        for hz in [500, 1000, 2000, 4000]:
            ear_df[f"ac_{hz}Hz"] = col_or_nan(df, f"{side}_{hz}Hz")
        ear_df["ac_PTA"] = col_or_nan(df, f"{side}_PTA")
        ear_frames.append(ear_df)
    return pd.concat(ear_frames, axis=0, ignore_index=True)


def build_task2_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ear_frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["hearing_type"] = col_or_nan(df, f"hearing_type_{side}")
        for hz in TASK2_EAR_LEVEL_AC_FREQS:
            ear_df[f"ac_{hz}Hz"] = col_or_nan(df, f"{side}_{hz}Hz")
            ear_df[f"ac_{hz}Hz_nr"] = col_or_nan(df, f"{side}_{hz}Hz_nr")
        for hz in TASK2_ABG_FREQS:
            ear_df[f"bc_{hz}Hz"] = col_or_nan(df, f"bc_{side}_{hz}Hz")
            ear_df[f"bc_{hz}Hz_nr"] = col_or_nan(df, f"bc_{side}_{hz}Hz_nr")
            ear_df[f"bc_{hz}Hz_missing"] = col_or_nan(df, f"bc_{side}_{hz}Hz_missing")
            ear_df[f"abg_{hz}Hz"] = col_or_nan(df, f"abg_{side}_{hz}Hz")
            ear_df[f"abg_{hz}Hz_missing"] = col_or_nan(df, f"abg_{side}_{hz}Hz_missing")
            ear_df[f"abg_{hz}Hz_censored"] = col_or_nan(df, f"abg_{side}_{hz}Hz_censored")
        ear_frames.append(ear_df)
    return pd.concat(ear_frames, axis=0, ignore_index=True)


def build_task3_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    name_map = {
        "Vea": "tymp_Vea",
        "peak_daPa": "tymp_peak_daPa",
        "peak_mmho": "tymp_peak_mmho",
        "Width_daPa": "tymp_Width_daPa",
    }
    ear_frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["tymp_type"] = col_or_nan(df, f"tymp_{side}_type")
        for suffix, neutral_col in name_map.items():
            src_col = f"tymp_{side}_{suffix}"
            ear_df[neutral_col] = col_or_nan(df, src_col)
            for flag in ["real_zero", "missing_zero", "np_zero"]:
                ear_df[f"{neutral_col}_{flag}"] = col_or_nan(df, f"{src_col}_{flag}")
        ear_frames.append(ear_df)
    return pd.concat(ear_frames, axis=0, ignore_index=True)


def filter_task2_standard_hearing_types(df: pd.DataFrame) -> pd.DataFrame:
    label = df["hearing_type"].astype(str).str.strip().str.upper()
    keep = label.isin(TASK2_STANDARD_HEARING_TYPES)
    out = df.loc[keep].reset_index(drop=True)
    out.attrs["task2_label_filter"] = {
        "allowed_labels": sorted(TASK2_STANDARD_HEARING_TYPES),
        "rows_before_filter": int(len(df)),
        "rows_after_filter": int(len(out)),
        "rows_excluded": int((~keep).sum()),
        "labels_before_filter": {
            str(k): int(v)
            for k, v in label.value_counts(dropna=False).sort_index().items()
        },
        "excluded_labels": {
            str(k): int(v)
            for k, v in label.loc[~keep].value_counts(dropna=False).sort_index().items()
        },
    }
    return out


def prepare_task1_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return shared_prepare_task1_dataframe(df)


def prepare_task2_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return shared_prepare_task2_dataframe(df)


def prepare_task3_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return shared_prepare_task3_dataframe(df)


def prepare_task_dataframe(task_name: str, raw: pd.DataFrame) -> pd.DataFrame:
    if task_name == "Task1":
        return prepare_task1_dataframe(raw)
    if task_name == "Task2":
        return prepare_task2_dataframe(raw)
    if task_name == "Task3":
        return prepare_task3_dataframe(raw)
    raise KeyError(f"Unknown task: {task_name}")


def pad_and_clean(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [sanitize_column_name(c) for c in out.columns]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def clean_label_columns(df: pd.DataFrame, label_cols: List[str]):
    out = df.copy()
    missing_stats = {}
    for lbl in label_cols:
        if lbl not in out.columns:
            raise KeyError(f"Missing label column: {lbl}")
        s = out[lbl].astype(str).str.strip()
        missing_lookup = {str(token).strip().lower() for token in MISSING_TOKENS}
        s = s.mask(s.str.lower().isin(missing_lookup), np.nan)
        out[lbl] = s
        missing_stats[lbl] = int(out[lbl].isna().sum())

    before = len(out)
    out = out.dropna(subset=label_cols).reset_index(drop=True)
    missing_stats["rows_before"] = int(before)
    missing_stats["rows_after"] = int(len(out))
    missing_stats["rows_dropped_any_label_missing"] = int(before - len(out))
    return out, missing_stats


def score_split_class_coverage(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_cols: List[str],
) -> Tuple[int, int, int, float]:
    train_missing = 0
    val_missing = 0
    val_singletons = 0
    distribution_gap = 0.0

    for lbl in label_cols:
        full_counts = full_df[lbl].value_counts()
        train_counts = train_df[lbl].value_counts()
        val_counts = val_df[lbl].value_counts()
        for cls, total in full_counts.items():
            train_count = int(train_counts.get(cls, 0))
            val_count = int(val_counts.get(cls, 0))
            if train_count == 0:
                train_missing += 1
            if total >= 2 and val_count == 0:
                val_missing += 1
            if total >= 3 and val_count == 1:
                val_singletons += 1
            full_ratio = float(total) / max(len(full_df), 1)
            val_ratio = float(val_count) / max(len(val_df), 1)
            distribution_gap += abs(full_ratio - val_ratio)
    return (train_missing, val_missing, val_singletons, distribution_gap)


def split_with_strategy(
    rest_df: pd.DataFrame,
    forced_df: pd.DataFrame,
    stratify_col: str,
    seed: int,
    group_col: str = "case_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    use_group_split = (
        group_col in rest_df.columns
        and rest_df[group_col].notna().all()
        and rest_df[group_col].nunique() >= 2
    )

    if use_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        groups = rest_df[group_col].astype(str)
        train_idx, val_idx = next(splitter.split(rest_df, groups=groups))
        train_df = rest_df.iloc[train_idx].copy()
        val_df = rest_df.iloc[val_idx].copy()
    elif len(rest_df) >= 10 and stratify_col is not None and (not rest_df[stratify_col].isna().any()):
        vc = rest_df[stratify_col].value_counts()
        if len(vc) > 1 and vc.min() >= 2:
            train_df, val_df = train_test_split(
                rest_df,
                test_size=0.2,
                random_state=seed,
                stratify=rest_df[stratify_col],
            )
        else:
            train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed)
    elif len(rest_df) >= 2:
        train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed)
    else:
        train_df = rest_df.copy()
        val_df = rest_df.iloc[:0].copy()

    train_df = pd.concat([train_df, forced_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def split_dataframe(df: pd.DataFrame, task_name: str, seed: int):
    label_cols = TASK_INFO[task_name]["label_cols"]
    strat_col = label_cols[0]
    group_col = "case_id"
    work = df.copy()
    rare_summary = {}
    force_train_idx = set()

    vc = work[strat_col].value_counts()
    rare_classes = [k for k, v in vc.items() if v < 2]
    for cls in rare_classes:
        idxs = work.index[work[strat_col] == cls].tolist()
        force_train_idx.update(idxs)
        rare_summary[str(cls)] = len(idxs)

    if group_col in work.columns and force_train_idx:
        forced_groups = set(work.loc[sorted(force_train_idx), group_col].astype(str).tolist())
        grouped_idxs = work.index[work[group_col].astype(str).isin(forced_groups)].tolist()
        force_train_idx.update(grouped_idxs)
        rare_summary["_forced_case_ids_count"] = int(len(forced_groups))

    forced_df = work.loc[sorted(list(force_train_idx))].copy()
    rest_df = work.drop(index=list(force_train_idx)).copy()

    candidates = []
    n_candidate_seeds = 200 if len(work) < 500 else 50
    for offset in range(n_candidate_seeds):
        candidate_seed = seed * 1000 + offset
        for candidate_strat_col in [strat_col, None] + [lbl for lbl in label_cols if lbl != strat_col]:
            train_df, val_df = split_with_strategy(rest_df, forced_df, candidate_strat_col, candidate_seed, group_col)
            score = score_split_class_coverage(work, train_df, val_df, label_cols)
            candidates.append((score, candidate_seed, candidate_strat_col, train_df, val_df))

    best_score, selected_seed, selected_strat_col, train_df, val_df = min(candidates, key=lambda x: x[0])
    rare_summary["_selected_candidate_seed"] = int(selected_seed)
    rare_summary["_selected_stratify_col"] = "random" if selected_strat_col is None else str(selected_strat_col)
    rare_summary["_split_score_train_missing"] = int(best_score[0])
    rare_summary["_split_score_val_missing"] = int(best_score[1])
    rare_summary["_split_score_val_singletons"] = int(best_score[2])
    rare_summary["_split_score_distribution_gap"] = float(best_score[3])
    if group_col in work.columns:
        train_groups = set(train_df[group_col].astype(str).tolist())
        val_groups = set(val_df[group_col].astype(str).tolist())
        rare_summary["_group_split_enabled"] = True
        rare_summary["_train_case_ids"] = int(len(train_groups))
        rare_summary["_val_case_ids"] = int(len(val_groups))
        rare_summary["_group_overlap_count"] = int(len(train_groups & val_groups))
    else:
        rare_summary["_group_split_enabled"] = False
    return train_df, val_df, rare_summary


def compute_norm_meta(train_df: pd.DataFrame, feature_cols: List[str]):
    mu = {}
    sigma = {}
    for col in feature_cols:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std(ddof=0))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        mu[col] = mean
        sigma[col] = std
    return {"mu": mu, "sigma": sigma}


def apply_norm(df: pd.DataFrame, feature_cols: List[str], mu: Dict[str, float], sigma: Dict[str, float]):
    out = df.copy()
    for col in feature_cols:
        out[col] = (out[col].astype(float) - mu[col]) / sigma[col]
    return out


def parse_seed_list(value: str) -> List[int]:
    seeds = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [0]


def parse_model_list(value: str) -> List[str]:
    if not value or value.lower() == "all":
        return [
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "mlp",
            "xgboost",
            "lightgbm",
            "catboost",
        ]
    return [part.strip() for part in value.split(",") if part.strip()]


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_feature_sets(data_dir: Path, feature_mode: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    raw_dfs = {}
    union_features = set()
    for task_name, info in TASK_INFO.items():
        csv_path = data_dir / info["csv"]
        raw = load_tabular_data(csv_path)
        prepared = prepare_task_dataframe(task_name, raw)
        raw_dfs[task_name] = prepared
        union_features.update(info["feature_cols"])

    if feature_mode == "union":
        features = sorted(union_features)
        feature_sets = {task_name: features for task_name in TASK_INFO}
    elif feature_mode == "task":
        feature_sets = {
            task_name: list(info["feature_cols"])
            for task_name, info in TASK_INFO.items()
        }
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return raw_dfs, feature_sets


def sample_key_series(df: pd.DataFrame) -> pd.Series:
    if "case_id" not in df.columns or "ear_side" not in df.columns:
        raise KeyError("Locked-test split requires case_id and ear_side columns.")
    return df["case_id"].astype(str) + "__" + df["ear_side"].astype(str)


def load_split_manifest(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"task", "seed", "split", "sample_key"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {sorted(missing)}")
    manifest["sample_key"] = manifest["sample_key"].astype(str)
    return manifest


def split_dataframe_from_manifest(
    df: pd.DataFrame,
    task_name: str,
    seed: int,
    split_manifest: pd.DataFrame,
    eval_mode: str,
):
    task_manifest = split_manifest[
        (split_manifest["task"].astype(str) == task_name)
        & (pd.to_numeric(split_manifest["seed"], errors="coerce") == int(seed))
    ].copy()
    if task_manifest.empty:
        raise ValueError(f"No split-manifest rows for task={task_name}, seed={seed}")

    if eval_mode == "locked_test":
        train_splits = {"train", "val"}
        eval_splits = {"test"}
    else:
        train_splits = {"train"}
        eval_splits = {"val"}

    work = df.copy()
    work["_sample_key"] = sample_key_series(work)
    train_keys = set(task_manifest.loc[task_manifest["split"].isin(train_splits), "sample_key"].astype(str))
    eval_keys = set(task_manifest.loc[task_manifest["split"].isin(eval_splits), "sample_key"].astype(str))
    train_df = work.loc[work["_sample_key"].isin(train_keys)].drop(columns=["_sample_key"]).copy()
    eval_df = work.loc[work["_sample_key"].isin(eval_keys)].drop(columns=["_sample_key"]).copy()
    if train_df.empty or eval_df.empty:
        raise ValueError(
            f"Manifest split produced empty train/eval data for task={task_name}, seed={seed}, eval_mode={eval_mode}"
        )

    overlap = train_keys & eval_keys
    split_meta = {
        "_split_source": "split_manifest",
        "_evaluation_mode": eval_mode,
        "_manifest_rows": int(len(task_manifest)),
        "_manifest_train_keys": int(len(train_keys)),
        "_manifest_eval_keys": int(len(eval_keys)),
        "_matched_train_rows": int(len(train_df)),
        "_matched_eval_rows": int(len(eval_df)),
        "_train_eval_key_overlap": int(len(overlap)),
    }
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True), split_meta

def prepare_split(
    raw_df: pd.DataFrame,
    task_name: str,
    feature_cols: List[str],
    seed: int,
    eval_mode: str = "grouped",
    split_manifest: pd.DataFrame | None = None,
):
    info = TASK_INFO[task_name]
    df = pad_and_clean(raw_df, feature_cols)
    df, missing_stats = clean_label_columns(df, info["label_cols"])
    if len(df) < 2:
        raise ValueError(f"{task_name} has too few rows after label cleaning: {len(df)}")

    if split_manifest is not None:
        train_df, val_df, rare_summary = split_dataframe_from_manifest(
            df,
            task_name,
            seed,
            split_manifest,
            "locked_test" if eval_mode == "locked_test" else "grouped",
        )
    else:
        train_df, val_df, rare_summary = split_dataframe(df, task_name, seed)
    norm = compute_norm_meta(train_df, feature_cols)
    train_df_n = apply_norm(train_df, feature_cols, norm["mu"], norm["sigma"])
    val_df_n = apply_norm(val_df, feature_cols, norm["mu"], norm["sigma"])

    lbl = info["label_cols"][0]
    encoder = LabelEncoder()
    encoder.fit(df[lbl].astype(str))
    y_train = encoder.transform(train_df[lbl].astype(str))
    y_val = encoder.transform(val_df[lbl].astype(str))
    x_train = train_df_n[feature_cols].astype(np.float32).to_numpy()
    x_val = val_df_n[feature_cols].astype(np.float32).to_numpy()

    meta = {
        "task": task_name,
        "label_col": lbl,
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "evaluation_mode": eval_mode,
        "eval_split_name": "test" if eval_mode == "locked_test" else "val",
        "class_names": [str(name) for name in encoder.classes_],
        "train_class_counts": {
            str(k): int(v)
            for k, v in train_df[lbl].astype(str).value_counts().sort_index().items()
        },
        "val_class_counts": {
            str(k): int(v)
            for k, v in val_df[lbl].astype(str).value_counts().sort_index().items()
        },
        "missing_stats": missing_stats,
        "rare_summary": rare_summary,
    }
    return x_train, y_train, x_val, y_val, meta


def optional_package_available(model_name: str) -> bool:
    package = OPTIONAL_MODEL_PACKAGES.get(model_name)
    return package is None or importlib.util.find_spec(package) is not None


def build_models(seed: int, quick: bool, requested_models: Iterable[str]):
    requested = set(requested_models)
    models = {}
    skipped = []

    if "logistic_regression" in requested:
        models["logistic_regression"] = {
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=1000 if quick else 5000,
                solver="lbfgs",
            ),
            "sample_weight": False,
        }

    if "random_forest" in requested:
        models["random_forest"] = {
            "estimator": RandomForestClassifier(
                n_estimators=80 if quick else 500,
                random_state=seed,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
                n_jobs=1,
            ),
            "sample_weight": False,
        }

    if "hist_gradient_boosting" in requested:
        models["hist_gradient_boosting"] = {
            "estimator": HistGradientBoostingClassifier(
                random_state=seed,
                max_iter=60 if quick else 250,
                learning_rate=0.08,
                l2_regularization=0.01,
            ),
            "sample_weight": True,
        }

    if "mlp" in requested:
        models["mlp"] = {
            "estimator": MLPClassifier(
                hidden_layer_sizes=(32,) if quick else (64, 32),
                activation="relu",
                alpha=1e-4,
                batch_size="auto",
                early_stopping=True,
                max_iter=80 if quick else 300,
                random_state=seed,
            ),
            "sample_weight": True,
        }

    if "xgboost" in requested:
        if optional_package_available("xgboost"):
            from xgboost import XGBClassifier

            models["xgboost"] = {
                "estimator": XGBClassifier(
                    n_estimators=80 if quick else 300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=seed,
                    eval_metric="mlogloss",
                ),
                "sample_weight": True,
            }
        else:
            skipped.append({"model": "xgboost", "reason": "package_not_installed"})

    if "lightgbm" in requested:
        if optional_package_available("lightgbm"):
            from lightgbm import LGBMClassifier

            models["lightgbm"] = {
                "estimator": LGBMClassifier(
                    n_estimators=80 if quick else 300,
                    learning_rate=0.05,
                    random_state=seed,
                    class_weight="balanced",
                    verbose=-1,
                ),
                "sample_weight": False,
            }
        else:
            skipped.append({"model": "lightgbm", "reason": "package_not_installed"})

    if "catboost" in requested:
        if optional_package_available("catboost"):
            from catboost import CatBoostClassifier

            models["catboost"] = {
                "estimator": CatBoostClassifier(
                    iterations=80 if quick else 300,
                    learning_rate=0.05,
                    depth=4,
                    random_seed=seed,
                    auto_class_weights="Balanced",
                    verbose=False,
                    loss_function="MultiClass",
                ),
                "sample_weight": False,
            }
        else:
            skipped.append({"model": "catboost", "reason": "package_not_installed"})

    known = {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "mlp",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    for model_name in sorted(requested - known):
        skipped.append({"model": model_name, "reason": "unknown_model"})

    return models, skipped


def fit_estimator(estimator, x_train, y_train, sample_weight=None):
    fit_params = inspect.signature(estimator.fit).parameters
    if sample_weight is not None and "sample_weight" in fit_params:
        return estimator.fit(x_train, y_train, sample_weight=sample_weight)
    return estimator.fit(x_train, y_train)


def evaluate_model(y_true, y_pred, class_names: List[str]) -> Tuple[Dict[str, float], List[Dict[str, object]], List[List[int]]]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    per_class_rows = []
    for class_name in class_names:
        row = report.get(class_name, {})
        per_class_rows.append({
            "class_name": class_name,
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        })
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics, per_class_rows, cm


def run_ml_baselines(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list(args.seeds)
    requested_models = parse_model_list(args.models)
    raw_dfs, feature_sets = build_feature_sets(data_dir, args.feature_mode)
    split_manifest = load_split_manifest(args.split_manifest) if args.eval_mode == "locked_test" else None
    if args.eval_mode == "locked_test" and split_manifest is None:
        raise ValueError("--split-manifest is required when --eval-mode locked_test")

    summary_rows = []
    per_class_rows = []
    confusion_matrices = []
    skipped_rows = []

    for seed in seeds:
        models, unavailable = build_models(seed, args.quick, requested_models)
        for skipped in unavailable:
            skipped_rows.append({"seed": seed, "task": "ALL", **skipped})

        for task_name in TASK_INFO:
            feature_cols = feature_sets[task_name]
            x_train, y_train, x_val, y_val, meta = prepare_split(
                raw_dfs[task_name],
                task_name,
                feature_cols,
                seed,
                eval_mode=args.eval_mode,
                split_manifest=split_manifest,
            )
            if len(np.unique(y_train)) < 2:
                for model_name in models:
                    skipped_rows.append({
                        "seed": seed,
                        "task": task_name,
                        "model": model_name,
                        "reason": "train_split_has_less_than_two_classes",
                    })
                continue

            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            for model_name, model_spec in models.items():
                estimator = model_spec["estimator"]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit_estimator(
                            estimator,
                            x_train,
                            y_train,
                            sample_weight=sample_weight if model_spec["sample_weight"] else None,
                        )
                    y_pred = estimator.predict(x_val)
                    metrics, task_per_class, cm = evaluate_model(
                        y_val,
                        y_pred,
                        meta["class_names"],
                    )
                except Exception as exc:
                    skipped_rows.append({
                        "seed": seed,
                        "task": task_name,
                        "model": model_name,
                        "reason": f"fit_or_eval_failed: {type(exc).__name__}: {exc}",
                    })
                    continue

                summary_rows.append({
                    "seed": seed,
                    "task": task_name,
                    "label_col": meta["label_col"],
                    "model": model_name,
                    "feature_mode": args.feature_mode,
                    "evaluation_mode": meta["evaluation_mode"],
                    "eval_split_name": meta["eval_split_name"],
                    "feature_count": meta["feature_count"],
                    "train_n": meta["train_n"],
                    "val_n": meta["val_n"],
                    "classes": json_dumps(meta["class_names"]),
                    "train_class_counts": json_dumps(meta["train_class_counts"]),
                    "val_class_counts": json_dumps(meta["val_class_counts"]),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "quick": bool(args.quick),
                })
                for row in task_per_class:
                    per_class_rows.append({
                        "seed": seed,
                        "task": task_name,
                        "label_col": meta["label_col"],
                        "model": model_name,
                        "feature_mode": args.feature_mode,
                        "evaluation_mode": meta["evaluation_mode"],
                        "eval_split_name": meta["eval_split_name"],
                        **row,
                    })
                confusion_matrices.append({
                    "seed": seed,
                    "task": task_name,
                    "label_col": meta["label_col"],
                    "model": model_name,
                    "feature_mode": args.feature_mode,
                    "evaluation_mode": meta["evaluation_mode"],
                    "eval_split_name": meta["eval_split_name"],
                    "class_names": meta["class_names"],
                    "matrix": cm,
                })

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    skipped_df = pd.DataFrame(skipped_rows, columns=["seed", "task", "model", "reason"])

    prefix = "ml_baseline_locked_test" if args.eval_mode == "locked_test" else "ml_baseline"
    summary_path = output_dir / f"{prefix}_summary.csv"
    per_class_path = output_dir / f"{prefix}_per_class.csv"
    summary_5seed_path = output_dir / f"{prefix}_summary_5seed.csv"
    per_class_5seed_path = output_dir / f"{prefix}_per_class_5seed.csv"
    skipped_path = output_dir / f"{prefix}_skipped_models.csv"
    cm_path = output_dir / f"{prefix}_confusion_matrices.json"
    manifest_path = output_dir / f"{prefix}_manifest.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    per_class_df.to_csv(per_class_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_5seed_path, index=False, encoding="utf-8-sig")
    per_class_df.to_csv(per_class_5seed_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    cm_path.write_text(json.dumps(confusion_matrices, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "models_requested": requested_models,
        "feature_mode": args.feature_mode,
        "evaluation_mode": args.eval_mode,
        "split_manifest": args.split_manifest if args.eval_mode == "locked_test" else None,
        "preprocessing_reference": "Mirrors train_v74.py ear-level preprocessing, grouped/locked split, and z-score normalization without importing torch.",
        "quick": bool(args.quick),
        "files": {
            "summary": str(summary_path),
            "per_class": str(per_class_path),
            "summary_5seed": str(summary_5seed_path),
            "per_class_5seed": str(per_class_5seed_path),
            "skipped_models": str(skipped_path),
            "confusion_matrices": str(cm_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary_df, skipped_df, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Run classical ML baselines with the v7.4 task preprocessing and grouped split."
    )
    parser.add_argument("--data-dir", default=".", help="Directory containing task CSV files.")
    parser.add_argument("--output-dir", default="results_v74/ml_baselines", help="Output directory.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2,3,4.")
    parser.add_argument(
        "--feature-mode",
        default="task",
        choices=["task", "union"],
        help="task = each task uses its own feature list; union = pad all tasks to union features.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or all. Supported: logistic_regression,random_forest,hist_gradient_boosting,mlp,xgboost,lightgbm,catboost.",
    )
    parser.add_argument("--eval-mode", default="grouped", choices=["grouped", "locked_test"], help="grouped uses the internal grouped validation split; locked_test trains on manifest train+val and evaluates on manifest test.")
    parser.add_argument("--split-manifest", default="", help="Split manifest CSV from split_protocol_v74.py; required for --eval-mode locked_test.")
    parser.add_argument("--quick", action="store_true", help="Use smaller models for smoke tests.")
    args = parser.parse_args()

    summary_df, skipped_df, manifest = run_ml_baselines(args)
    if summary_df.empty:
        print("No baseline model finished successfully.")
    else:
        cols = ["seed", "task", "model", "feature_mode", "accuracy", "macro_f1", "balanced_accuracy"]
        print(summary_df[cols].to_string(index=False))
    if not skipped_df.empty:
        print("\nSkipped models:")
        print(skipped_df.to_string(index=False))
    print(f"\nSaved summary: {manifest['files']['summary']}")
    print(f"Saved per-class: {manifest['files']['per_class']}")
    print(f"Saved confusion matrices: {manifest['files']['confusion_matrices']}")


if __name__ == "__main__":
    main()
