import argparse
import copy
import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from clinical_rules_v74 import (
    MISSING_TOKENS,
    TASK2_ABG_FREQS,
    TASK2_AC_NR_LIMITS_DB,
    TASK2_BC_NR_LIMITS_DB,
    TASK2_EAR_LEVEL_AC_FREQS,
    TASK2_STANDARD_HEARING_TYPES,
    infer_task2_hearing_type_from_rule as shared_infer_task2_hearing_type_from_rule,
    rule_decision,
    score_rule_decision,
    score_task3_expert_consistency,
)
from mtl_meta_irl_transformer import MTLMetaIRLTransformer
from preprocessing_v74 import (
    load_tabular_data,
    prepare_task1_dataframe as shared_prepare_task1_dataframe,
    prepare_task2_dataframe as shared_prepare_task2_dataframe,
    prepare_task3_dataframe as shared_prepare_task3_dataframe,
)


# 允許使用的 GPU Compute Capability 清單
# 不在清單內時，自動改用 CPU
SUPPORTED_CC = {(5,0),(6,0),(6,1),(7,0),(7,5),(8,0),(8,6),(9,0),(8,9)}


def safe_pick_device():
    """安全選擇訓練裝置。"""
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) not in SUPPORTED_CC:
                print(f"[MetaIRL v7.4] CUDA sm_{major}{minor} not fully supported, fallback to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
    except Exception as e:
        print(f"[MetaIRL v7.4] safe_pick_device error: {e}")
    return torch.device("cpu")


# 全域裝置
DEVICE = safe_pick_device()

# ===== 訓練超參數 =====
LR = 5e-4  # 學習率
BATCH = 32  # batch size
EPOCHS = 50  # 最大訓練 epoch
PATIENCE = 10  # early stopping 容忍次數
WEIGHT_DECAY = 1e-4
DEFAULT_REWARD_WEIGHT = 0.20  # IRL reward loss 的權重
DEFAULT_TASK2_UNCERTAIN_REWARD_CONFIDENCE = 0.50
DEFAULT_PARTIAL_RULE_REWARD_CONFIDENCE = 0.50
DEFAULT_REWARD_CONFIDENCE_FLOOR = 0.25
GRAD_CLIP = 1.0  # gradient clipping 上限
LOW_SUPPORT_THRESHOLD = 5
DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS = {
    "Task1": {
        "mask_pta": 0.35,
        "mask_high_freq": 0.25,
        "mask_low_freq": 0.20,
        "mask_all_ac": 0.20,
    },
    "Task2": {
        "no_bc": 0.50,
        "partial_bc": 0.30,
        "multi_ac_nr": 0.20,
    },
    "Task3": {
        "no_peak": 0.20,
        "no_width": 0.15,
        "np_like": 0.25,
        "no_tymp": 0.40,
    },
}
DEFAULT_MODEL_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dropout": 0.15,
    "proto_alpha": 0.35,
    "proto_temperature": 1.0,
}
# NOTE 2026-05-20 P2-14:
# 問題：原本每個 epoch 依序跑完整 Task1/Task2/Task3 loader，Task1 樣本數最大，容易主導 shared encoder。
# 用法：run_experiment() 會讀取 task_sampling_mode；預設 equal_steps 代表每個 task 每個 epoch 跑相同 steps。
# 例：Task1 loader=3 steps、Task2=1、Task3=1 時，equal_steps 會讓三個 task 都跑 3 steps，Task2/Task3 會循環抽 batch。
DEFAULT_TASK_SAMPLING_CONFIG = {
    "task_sampling_mode": "equal_steps",
    "steps_per_task": None,
}
# NOTE 2026-05-20 P2-12:
# 問題：舊 meta 流程把同一個 batch 同時當 support 與 query，validation 又沒有測 support，meta 效果不乾淨。
# 用法：support_per_class 控制 build_episode() 每類最多取幾筆 support；validate_with_meta_support 控制 validation 是否也切 support/query。
# 例：support_per_class=2 時，同一類若有 5 筆，最多 2 筆做 prototype support，其餘 query 才計算 CE/reward loss。
DEFAULT_META_EPISODE_CONFIG = {
    "support_per_class": 2,
    "validate_with_meta_support": True,
}

SINGLE_TASK_TARGETS = ["Task1", "Task2", "Task3"]
SINGLE_TASK_EXPERIMENT_ALIASES = {
    "single_task_task1": "Task1",
    "single_task_task2": "Task2",
    "single_task_task3": "Task3",
}
EXPERIMENT_CHOICES = [
    "full",
    "no_meta",
    "no_irl",
    "single_task",
    *SINGLE_TASK_EXPERIMENT_ALIASES.keys(),
]

MODEL_SIZE_CONFIGS = {
    "base": {
        **DEFAULT_MODEL_CONFIG,
    },
    "small": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.15,
        "proto_alpha": 0.25,
        "proto_temperature": 1.5,
    },
    "tiny": {
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dropout": 0.10,
        "proto_alpha": 0.20,
        "proto_temperature": 1.5,
    },
}

MISSING_AUG_PROFILES = {
    "m05_balanced": {
        "missing_aug_p": 0.05,
        "missing_aug_strategy_weights": DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS,
    },
    "m10_balanced": {
        "missing_aug_p": 0.10,
        "missing_aug_strategy_weights": DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS,
    },
    "m15_bc_dominant": {
        "missing_aug_p": 0.15,
        "missing_aug_strategy_weights": {
            "Task1": DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS["Task1"],
            "Task2": {"no_bc": 0.65, "partial_bc": 0.25, "multi_ac_nr": 0.10},
            "Task3": DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS["Task3"],
        },
    },
    "m15_tymp_dominant": {
        "missing_aug_p": 0.15,
        "missing_aug_strategy_weights": {
            "Task1": DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS["Task1"],
            "Task2": DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS["Task2"],
            "Task3": {"no_peak": 0.10, "no_width": 0.05, "np_like": 0.25, "no_tymp": 0.60},
        },
    },
    "m20_heavy_balanced": {
        "missing_aug_p": 0.20,
        "missing_aug_strategy_weights": {
            "Task1": {"mask_pta": 0.25, "mask_high_freq": 0.25, "mask_low_freq": 0.20, "mask_all_ac": 0.30},
            "Task2": {"no_bc": 0.55, "partial_bc": 0.25, "multi_ac_nr": 0.20},
            "Task3": {"no_peak": 0.15, "no_width": 0.10, "np_like": 0.25, "no_tymp": 0.50},
        },
    },
    "m30_stress": {
        "missing_aug_p": 0.30,
        "missing_aug_strategy_weights": {
            "Task1": {"mask_pta": 0.20, "mask_high_freq": 0.20, "mask_low_freq": 0.20, "mask_all_ac": 0.40},
            "Task2": {"no_bc": 0.70, "partial_bc": 0.20, "multi_ac_nr": 0.10},
            "Task3": {"no_peak": 0.10, "no_width": 0.05, "np_like": 0.25, "no_tymp": 0.60},
        },
    },
}


def missing_aug_profile(profile_name: str) -> Dict[str, object]:
    profile = copy.deepcopy(MISSING_AUG_PROFILES[profile_name])
    profile["missing_aug_profile"] = profile_name
    return profile


def make_run_config(
    name: str,
    model_size: str,
    missing_profile: str,
    reward_weight: float,
    *,
    task_sampling_mode: str = "round_robin_k1",
    steps_per_task: int = None,
    support_per_class: int = 1,
    validate_with_meta_support: bool = True,
) -> Dict[str, object]:
    return {
        "name": name,
        "lr": 5e-4,
        "batch": 512,
        "epochs": 80,
        "patience": 15,
        "weight_decay": 1e-4,
        "reward_weight": float(reward_weight),
        **copy.deepcopy(MODEL_SIZE_CONFIGS[model_size]),
        **missing_aug_profile(missing_profile),
        "task_sampling_mode": task_sampling_mode,
        "steps_per_task": steps_per_task,
        "support_per_class": support_per_class,
        "validate_with_meta_support": validate_with_meta_support,
    }


RUN_CONFIGS = [
    make_run_config("run_01_base_m05_balanced_r015", "base", "m05_balanced", 0.15),
    make_run_config("run_02_base_m10_balanced_r015", "base", "m10_balanced", 0.15),
    make_run_config("run_03_base_m15_bc_dominant_r015", "base", "m15_bc_dominant", 0.15),
    make_run_config("run_04_base_m15_tymp_dominant_r015", "base", "m15_tymp_dominant", 0.15),
    make_run_config("run_05_base_m20_heavy_balanced_r010", "base", "m20_heavy_balanced", 0.10),
    make_run_config("run_06_small_m05_balanced_r015", "small", "m05_balanced", 0.15),
    make_run_config("run_07_small_m10_balanced_r015", "small", "m10_balanced", 0.15),
    make_run_config("run_08_small_m15_bc_dominant_r015", "small", "m15_bc_dominant", 0.15),
    make_run_config("run_09_small_m15_tymp_dominant_r015", "small", "m15_tymp_dominant", 0.15),
    make_run_config("run_10_small_m20_heavy_balanced_r010", "small", "m20_heavy_balanced", 0.10),
    make_run_config("run_11_tiny_m05_balanced_r010", "tiny", "m05_balanced", 0.10),
    make_run_config("run_12_tiny_m10_balanced_r010", "tiny", "m10_balanced", 0.10),
    make_run_config("run_13_tiny_m15_bc_dominant_r010", "tiny", "m15_bc_dominant", 0.10),
    make_run_config("run_14_tiny_m15_tymp_dominant_r010", "tiny", "m15_tymp_dominant", 0.10),
    make_run_config("run_15_tiny_m20_heavy_balanced_r005", "tiny", "m20_heavy_balanced", 0.05),
]

ABLATION_RUN_CONFIGS = [
    make_run_config(
        "ablation_base_m10_equal_steps_r015",
        "base",
        "m10_balanced",
        0.15,
        task_sampling_mode="equal_steps",
    ),
    make_run_config(
        "ablation_base_m10_rr_k4_r015",
        "base",
        "m10_balanced",
        0.15,
        task_sampling_mode="round_robin_k4",
    ),
    make_run_config(
        "ablation_base_m10_support2_r015",
        "base",
        "m10_balanced",
        0.15,
        support_per_class=2,
    ),
    make_run_config(
        "ablation_base_m20_high_reward_r020",
        "base",
        "m20_heavy_balanced",
        0.20,
    ),
    make_run_config(
        "ablation_base_m30_stress_r010",
        "base",
        "m30_stress",
        0.10,
    ),
]

DEFAULT_RUN_CONFIG_FILE = Path("configs/run_configs_v74.json")
CONFIG_PRESET_TO_SPEC_KEY = {
    "recommended": "run_configs",
    "ablation": "ablation_run_configs",
}
CONFIG_ENTRY_KEYS = {"name", "model_size", "missing_profile", "reward_weight"}


SUPPORTED_RUN_CONFIG_SCHEMAS = {"v1", "v1_split"}


def resolve_run_config_file(path_value: str, base_dir: Path | None = None) -> Path:
    path = Path(path_value or DEFAULT_RUN_CONFIG_FILE)
    if not path.is_absolute():
        root = base_dir if base_dir is not None else Path(__file__).resolve().parent
        path = root / path
    return path.resolve()


def load_run_config_spec(path_value: str, base_dir: Path | None = None) -> dict:
    config_path = resolve_run_config_file(path_value, base_dir=base_dir)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Run config file not found: {config_path}. "
            "Expected JSON such as configs/run_configs_v74.json or a v1_split JSON."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict):
        raise ValueError(f"Run config file must contain a JSON object: {config_path}")
    schema_version = str(spec.get("schema_version"))
    if schema_version not in SUPPORTED_RUN_CONFIG_SCHEMAS:
        raise ValueError(f"Unsupported run config schema_version in {config_path}: {spec.get('schema_version')}")
    spec["_config_path"] = str(config_path)
    spec["_schema_version"] = schema_version
    return spec


def _require_mapping(spec: dict, key: str) -> dict:
    value = spec.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Run config JSON key '{key}' must be a non-empty object.")
    return value


def _require_list(spec: dict, key: str, expected_len: int) -> list:
    value = spec.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Run config JSON key '{key}' must be a list.")
    if len(value) != expected_len:
        raise ValueError(f"Run config JSON key '{key}' must contain {expected_len} entries, got {len(value)}.")
    return value


def _validate_model_size_configs(model_size_configs: dict) -> None:
    required = {"d_model", "nhead", "num_layers", "dropout", "proto_alpha", "proto_temperature"}
    for name, cfg in model_size_configs.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"model_size_configs.{name} must be an object.")
        missing = sorted(required - set(cfg))
        if missing:
            raise ValueError(f"model_size_configs.{name} missing fields: {missing}")
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(f"model_size_configs.{name} has invalid d_model/nhead: {d_model}/{nhead}")


def _validate_missing_aug_profiles(missing_aug_profiles: dict) -> None:
    for name, profile in missing_aug_profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"missing_aug_profiles.{name} must be an object.")
        p = float(profile.get("missing_aug_p", -1.0))
        if p <= 0.0:
            raise ValueError(f"missing_aug_profiles.{name}.missing_aug_p must be > 0, got {p}")
        weights = profile.get("missing_aug_strategy_weights")
        if not isinstance(weights, dict):
            raise ValueError(f"missing_aug_profiles.{name}.missing_aug_strategy_weights must be an object.")
        for task_name in ("Task1", "Task2", "Task3"):
            task_weights = weights.get(task_name)
            if not isinstance(task_weights, dict) or not task_weights:
                raise ValueError(f"missing_aug_profiles.{name} missing weights for {task_name}.")
            values = [float(v) for v in task_weights.values()]
            if any(v < 0.0 for v in values) or sum(values) <= 0.0:
                raise ValueError(f"missing_aug_profiles.{name}.{task_name} weights must be non-negative and sum > 0.")


def _default_run_values(spec: dict) -> dict:
    defaults = {
        "lr": 5e-4,
        "batch": 512,
        "epochs": 80,
        "patience": 15,
        "weight_decay": 1e-4,
        "task_sampling_mode": "round_robin_k1",
        "steps_per_task": None,
        "support_per_class": 1,
        "validate_with_meta_support": True,
    }
    configured = spec.get("default_run_values", {})
    if configured is not None:
        if not isinstance(configured, dict):
            raise ValueError("Run config JSON key 'default_run_values' must be an object when provided.")
        defaults.update(copy.deepcopy(configured))
    return defaults


def build_run_config_from_spec_entry(
    entry: dict,
    model_size_configs: dict,
    missing_aug_profiles: dict,
    default_values: dict,
) -> dict:
    if not isinstance(entry, dict):
        raise ValueError("Each run config entry must be an object.")
    missing = sorted(CONFIG_ENTRY_KEYS - set(entry))
    if missing:
        raise ValueError(f"Run config entry missing fields {missing}: {entry}")

    model_size = str(entry["model_size"])
    missing_profile = str(entry["missing_profile"])
    if model_size not in model_size_configs:
        raise ValueError(f"Unknown model_size '{model_size}' in run config '{entry.get('name')}'.")
    if missing_profile not in missing_aug_profiles:
        raise ValueError(f"Unknown missing_profile '{missing_profile}' in run config '{entry.get('name')}'.")

    cfg = copy.deepcopy(default_values)
    cfg.update(copy.deepcopy(model_size_configs[model_size]))
    profile = copy.deepcopy(missing_aug_profiles[missing_profile])
    profile["missing_aug_profile"] = missing_profile
    cfg.update(profile)
    cfg.update({
        "name": str(entry["name"]),
        "model_size": model_size,
        "missing_profile": missing_profile,
        "reward_weight": float(entry["reward_weight"]),
    })
    for key, value in entry.items():
        if key not in CONFIG_ENTRY_KEYS:
            cfg[key] = copy.deepcopy(value)

    d_model = int(cfg.get("d_model", DEFAULT_MODEL_CONFIG["d_model"]))
    nhead = int(cfg.get("nhead", DEFAULT_MODEL_CONFIG["nhead"]))
    if nhead <= 0 or d_model % nhead != 0:
        raise ValueError(f"Run config '{cfg['name']}' has invalid d_model/nhead: {d_model}/{nhead}")
    if float(cfg.get("missing_aug_p", 0.0) or 0.0) <= 0.0:
        raise ValueError(f"Run config '{cfg['name']}' must use missing_aug_p > 0.")
    return cfg


def _require_name_list(spec: dict, key: str) -> list[str]:
    value = spec.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Run config split JSON key '{key}' must be a non-empty list.")
    names = [str(item).strip() for item in value]
    if any(not name for name in names):
        raise ValueError(f"Run config split JSON key '{key}' contains an empty config name.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Run config split JSON key '{key}' contains duplicate names: {duplicates}")
    return names


def _validate_unique_run_config_names(selected: dict) -> None:
    names = [cfg["name"] for configs in (selected["recommended"], selected["ablation"]) for cfg in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Run config names must be unique. Duplicates: {duplicates}")


def _filter_run_config_sets_for_split(base_selected: dict, split_spec: dict) -> dict:
    include_names = _require_name_list(split_spec, "include_config_names")
    all_configs = base_selected["all"]
    name_to_cfg = {cfg["name"]: cfg for cfg in all_configs}
    missing = [name for name in include_names if name not in name_to_cfg]
    if missing:
        raise ValueError(f"Split run config references unknown config names: {missing}")

    expected_total = split_spec.get("expected_total_configs")
    if expected_total is not None and int(expected_total) != len(include_names):
        raise ValueError(
            f"Split expected_total_configs={expected_total} but include_config_names has {len(include_names)} entries."
        )

    recommended_names = {cfg["name"] for cfg in base_selected["recommended"]}
    ablation_names = {cfg["name"] for cfg in base_selected["ablation"]}
    part_name = str(split_spec.get("part_name") or Path(split_spec["_config_path"]).stem)
    split_config_path = str(split_spec["_config_path"])

    selected_all = []
    for name in include_names:
        cfg = copy.deepcopy(name_to_cfg[name])
        cfg["distributed_split_part"] = part_name
        cfg["distributed_split_config"] = split_config_path
        selected_all.append(cfg)

    selected = {
        "recommended": [cfg for cfg in selected_all if cfg["name"] in recommended_names],
        "ablation": [cfg for cfg in selected_all if cfg["name"] in ablation_names],
        "all": selected_all,
    }
    if not selected["all"]:
        raise ValueError(f"Split run config contains no selected configs: {split_config_path}")
    _validate_unique_run_config_names(selected)
    return selected


def load_run_config_sets(config_path: str) -> tuple[dict, str]:
    spec = load_run_config_spec(config_path)
    if spec.get("_schema_version") == "v1_split":
        base_config_file = spec.get("base_config_file")
        if not isinstance(base_config_file, str) or not base_config_file.strip():
            raise ValueError("Run config split JSON must define non-empty base_config_file.")
        split_path = Path(spec["_config_path"])
        base_config_path = resolve_run_config_file(base_config_file, base_dir=split_path.parent)
        base_selected, base_loaded_path = load_run_config_sets(str(base_config_path))
        selected = _filter_run_config_sets_for_split(base_selected, spec)
        return selected, f"{split_path} -> {base_loaded_path}"

    model_size_configs = _require_mapping(spec, "model_size_configs")
    missing_aug_profiles = _require_mapping(spec, "missing_aug_profiles")
    _validate_model_size_configs(model_size_configs)
    _validate_missing_aug_profiles(missing_aug_profiles)
    default_values = _default_run_values(spec)

    selected = {}
    for preset, key in CONFIG_PRESET_TO_SPEC_KEY.items():
        expected_len = 15 if preset == "recommended" else 5
        selected[preset] = [
            build_run_config_from_spec_entry(entry, model_size_configs, missing_aug_profiles, default_values)
            for entry in _require_list(spec, key, expected_len)
        ]
    selected["all"] = [*selected["recommended"], *selected["ablation"]]

    _validate_unique_run_config_names(selected)
    return selected, str(spec["_config_path"])


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

TASK3_NP_EXTREME_VALUES = {
    "tymp_right_peak_daPa": -999.0,
    "tymp_left_peak_daPa": -999.0,
    "tymp_right_peak_mmho": -1.0,
    "tymp_left_peak_mmho": -1.0,
}


# 三個任務各自的：
# - CSV 檔名
# - 標籤欄位
# - 原始特徵欄位
TASK_INFO = {
    "Task1": {
        "csv": "task1_all_three_common14_v1.csv",
        "label_cols": [
            "hearing_degree_WHO_PTAbased",
        ],
        "feature_cols": TASK1_EAR_LEVEL_FEATURES,
    },
    "Task2": {
        "csv": "task2_3_pure_data(6_24).xlsx",
        "label_cols": [
            "hearing_type",
        ],
        "feature_cols": TASK2_EAR_LEVEL_FEATURES,
    },
    "Task3": {
        "csv": "task2_3_pure_data(6_24).xlsx",
        "label_cols": [
            "tymp_type",
        ],
        "feature_cols": TASK3_EAR_LEVEL_FEATURES,
    },
}

'''
# ===== Dataset 定義 =====
# 將表格資料整理成 PyTorch 可讀取形式
class MultiLabelTabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 取出特徵向量
        x = row[self.feature_cols].astype(np.float32).values

        # 每個 label 欄位各自轉成整數類別
        ys = [int(row[lbl]) for lbl in self.label_cols]

        # raw 保留原始列資料，後面 reward 規則會用到
        raw = row.to_dict()
        return torch.tensor(x, dtype=torch.float32), [torch.tensor(y, dtype=torch.long) for y in ys], raw
'''

#############################################################
class MultiLabelTabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_cols: List[str],
        raw_df: pd.DataFrame = None,
    ):
        self.df = df.reset_index(drop=True)
        self.raw_df = raw_df.reset_index(drop=True) if raw_df is not None else self.df
        self.feature_cols = feature_cols
        self.label_cols = label_cols

        # 確保標準化資料與原始資料 row 對得上
        if len(self.df) != len(self.raw_df):
            raise ValueError("df and raw_df must have the same number of rows")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_row = self.raw_df.iloc[idx]

        # 模型輸入：使用標準化後 feature
        x = row[self.feature_cols].astype(np.float32).values

        # label：使用 LabelEncoder 後的整數 label
        ys = [int(row[lbl]) for lbl in self.label_cols]

        # reward 規則：使用原始尺度 feature，例如 dB、daPa、mmho
        raw = raw_row.to_dict()

        return torch.tensor(x, dtype=torch.float32), [torch.tensor(y, dtype=torch.long) for y in ys], raw

#############################################################

def collate_with_raw(batch):
    """自訂 DataLoader 的 collate function。

    將 batch 中的：
    - x 疊成 tensor
    - 多個 label 分欄堆疊
    - raw dict 保留成 list
    """
    xs = torch.stack([item[0] for item in batch], dim=0)
    n_labels = len(batch[0][1])
    ys = [torch.stack([item[1][j] for item in batch], dim=0) for j in range(n_labels)]
    raws = [item[2] for item in batch]
    return xs, ys, raws


def seed_everything(seed: int):
    """固定所有隨機種子，提升實驗可重現性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_column_name(x: str) -> str:
    """清理欄位名稱：去空白、空格改底線。"""
    x = str(x).strip()
    x = re.sub(r"\s+", "_", x)
    return x


def hz_from_col(col: str):
    """從欄位名稱抓頻率，例如 right_8000Hz / bc_left_500Hz -> 8000 / 500。"""
    m = re.search(r"_(250|500|750|1000|1500|2000|3000|4000|6000|8000)Hz$", str(col))
    return int(m.group(1)) if m else None


def parse_numeric_value(value, hz=None, nr_limits: Dict[int, float] = None):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    raw = str(value).strip()
    if raw in MISSING_TOKENS or raw.lower() in {str(token).lower() for token in MISSING_TOKENS}:
        return None
    upper = raw.upper().replace(" ", "")
    is_nr = upper.endswith("NR")
    cleaned = upper[:-2] if is_nr else upper
    cleaned = cleaned.replace("DB", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        numeric = float(match.group(0))
    except ValueError:
        return None
    if not np.isfinite(numeric):
        return None
    if is_nr and hz is not None and nr_limits and hz in nr_limits:
        return float(nr_limits[hz])
    return numeric

def clean_value(v, hz=None, nr_limits: Dict[int, float] = None):
    """安全轉 float；缺失、NaN、inf、不可轉換時統一補 0.0。"""
    x = parse_numeric_value(v, hz=hz, nr_limits=nr_limits)
    return 0.0 if x is None else x


def numeric_series(df: pd.DataFrame, col: str, nr_limits: Dict[int, float] = None) -> pd.Series:
    """將 DataFrame 欄位轉成數值 Series；nr_limits 只在 AC NR 轉上限時使用。"""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    hz = hz_from_col(col)
    parsed = df[col].map(lambda v: parse_numeric_value(v, hz=hz, nr_limits=nr_limits))
    return pd.to_numeric(parsed, errors="coerce")


def is_nr_value(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    upper = str(value).strip().upper().replace(" ", "")
    upper = upper.replace("DB", "")
    return re.fullmatch(r"-?\d+(?:\.\d+)?NR", upper) is not None

def nr_indicator_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[col].map(lambda v: 1.0 if is_nr_value(v) else 0.0).astype(float)


def row_numeric_value(row: dict, col: str, nr_limits: Dict[int, float] = None):
    """從 raw row 取單一欄位數值；與 DataFrame 前處理共用同一套 NR 邏輯。"""
    return parse_numeric_value(row.get(col), hz=hz_from_col(col), nr_limits=nr_limits)


def mean_present(values):
    """只平均實際存在的數值；若整組都缺失則回傳 None。"""
    vals = [x for x in values if x is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def convert_task2_nr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """轉換 Task2 NR 欄位，避免 NR 在 pad_and_clean() 被補成 0。

    AC 與 BC 的 NR 分別依使用者提供的各頻率機器上限轉換；
    例如 AC 95NR 依頻率轉上限，BC 75NR 依 BC 上限表轉換。
    """
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
    """集中計算 Task2 四頻率 ABG 與 missing/censored mask。"""
    derived_cols = {}
    for side in ["right", "left"]:
        for hz in TASK2_ABG_FREQS:
            ac_col = f"{side}_{hz}Hz"
            bc_col = f"bc_{side}_{hz}Hz"
            ac = numeric_series(df, ac_col, TASK2_AC_NR_LIMITS_DB)
            bc = numeric_series(df, bc_col, TASK2_BC_NR_LIMITS_DB)
            ac_nr = pd.to_numeric(df.get(f"{ac_col}_nr", 0.0), errors="coerce").fillna(0.0)
            bc_nr = pd.to_numeric(df.get(f"{bc_col}_nr", 0.0), errors="coerce").fillna(0.0)

            # missing mask 保留「原始 BC 缺測」資訊；BC NR 轉為上限值，不視為缺失。
            derived_cols[f"{bc_col}_missing"] = bc.isna().astype(float)
            abg = ac - bc
            derived_cols[f"abg_{side}_{hz}Hz"] = abg
            derived_cols[f"abg_{side}_{hz}Hz_missing"] = (ac.isna() | bc.isna()).astype(float)
            abg_censored = ((ac_nr >= 0.5) | (bc_nr >= 0.5)) & ~(ac.isna() | bc.isna())
            derived_cols[f"abg_{side}_{hz}Hz_censored"] = abg_censored.astype(float)

    return pd.concat([df, pd.DataFrame(derived_cols, index=df.index)], axis=1)


def col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """取欄位；若不存在則回傳同長度 NaN，方便建立 ear-level 表。"""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    """保留原始 row 群組，避免左右耳拆分後 train/val 洩漏。"""
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = np.arange(len(df))
    return df


def build_task1_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """將 Task1 row-level 左右耳資料拆成 ear-level sample。"""
    ear_frames = []
    for side in ["right", "left"]:
        ear_df = pd.DataFrame(index=df.index)
        ear_df["case_id"] = col_or_nan(df, "case_id")
        ear_df["ear_side"] = side
        ear_df["hearing_degree_WHO_PTAbased"] = col_or_nan(
            df,
            f"hearing_degree_WHO_PTAbased_{side}",
        )

        for hz in [500, 1000, 2000, 4000]:
            ear_df[f"ac_{hz}Hz"] = col_or_nan(df, f"{side}_{hz}Hz")
        ear_df["ac_PTA"] = col_or_nan(df, f"{side}_PTA")
        ear_frames.append(ear_df)

    return pd.concat(ear_frames, axis=0, ignore_index=True)


def build_task2_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """將 Task2 row-level 左右耳資料拆成 ear-level sample。"""
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


def filter_task2_standard_hearing_types(df: pd.DataFrame) -> pd.DataFrame:
    """Task2 只保留標準 hearing type，排除 high/low tone loss 等型態標籤。"""
    if "hearing_type" not in df.columns:
        return df
    label = df["hearing_type"].astype(str).str.strip().str.upper()
    keep = label.isin(TASK2_STANDARD_HEARING_TYPES)
    excluded = label.loc[~keep]
    out = df.loc[keep].reset_index(drop=True)
    out.attrs["task2_label_filter"] = {
        "allowed_labels": sorted(TASK2_STANDARD_HEARING_TYPES),
        "rows_before_filter": int(len(df)),
        "rows_after_filter": int(len(out)),
        "rows_excluded": int((~keep).sum()),
        "labels_before_filter": {str(k): int(v) for k, v in label.value_counts(dropna=False).sort_index().items()},
        "excluded_labels": {str(k): int(v) for k, v in excluded.value_counts(dropna=False).sort_index().items()},
    }
    return out


def build_task3_ear_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """將 Task3 row-level 左右耳 tympanogram 資料拆成 ear-level sample。"""
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


def prepare_task1_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """整理 Task1：清欄名、保留 case_id，並拆成單耳訓練資料。"""
    return shared_prepare_task1_dataframe(df)


def prepare_task2_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """整理 Task2：清欄名、NR 轉換、AC/BC/ABG 計算、單耳拆分與 label 過濾。"""
    return shared_prepare_task2_dataframe(df)


def prepare_task3_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """整理 Task3：清欄名、zero/NP 來源標記，並拆成單耳訓練資料。

    數值欄位會保留原始量尺；額外旗標讓模型區分真 0、空值與 NP。
    peak/compliance 的 NP 也會轉成極端 sentinel，避免被當成正常 0。
    """
    return shared_prepare_task3_dataframe(df)


def pad_and_clean(df: pd.DataFrame, union_features: List[str]) -> pd.DataFrame:
    """將資料表補齊到所有任務共用的 union_features。

    做的事情：
    1. 清理欄名
    2. 不存在的特徵欄位補 0.0
    3. 所有特徵欄位轉成數值，錯誤值補 0.0
    """
    df = df.copy()
    df.columns = [sanitize_column_name(c) for c in df.columns]
    for c in union_features:
        if c not in df.columns:
            df[c] = 0.0
    for c in union_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def clean_label_columns(df: pd.DataFrame, label_cols: List[str]):
    """清理標籤欄位並移除缺標籤的列。"""
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


def summarize_label_support(
    df: pd.DataFrame,
    label_cols: List[str],
    min_support: int = LOW_SUPPORT_THRESHOLD,
) -> Dict[str, Dict[str, object]]:
    """Summarize class counts and highlight classes below the support threshold."""
    summary = {}
    for lbl in label_cols:
        if lbl not in df.columns:
            continue
        counts = df[lbl].astype(str).value_counts(dropna=False).sort_index()
        counts_dict = {str(cls): int(count) for cls, count in counts.items()}
        low_support = {
            cls: count
            for cls, count in counts_dict.items()
            if count < min_support
        }
        summary[lbl] = {
            "counts": counts_dict,
            "low_support_threshold": int(min_support),
            "low_support_classes": low_support,
        }
    return summary


def build_class_support_report(
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_cols: List[str],
    min_support: int = LOW_SUPPORT_THRESHOLD,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Create all/train/val support reports before labels are encoded."""
    return {
        "all": summarize_label_support(all_df, label_cols, min_support),
        "train": summarize_label_support(train_df, label_cols, min_support),
        "val": summarize_label_support(val_df, label_cols, min_support),
    }


def split_dataframe_first_label_once(df: pd.DataFrame, task_name: str, seed: int):
    """切分 train / val，並保護稀有類別不被切到驗證集消失。"""
    label_cols = TASK_INFO[task_name]["label_cols"]
    strat_col = label_cols[0]  # 用第一個 label 做 stratify 參考
    work = df.copy()
    rare_summary = {}
    force_train_idx = set()

    # 找出樣本數 < 2 的罕見類別，強制放進 train
    vc = work[strat_col].value_counts()
    rare_classes = [k for k, v in vc.items() if v < 2]
    for cls in rare_classes:
        idxs = work.index[work[strat_col] == cls].tolist()
        force_train_idx.update(idxs)
        rare_summary[str(cls)] = len(idxs)

    forced_df = work.loc[sorted(list(force_train_idx))].copy()
    rest_df = work.drop(index=list(force_train_idx)).copy()

    # 優先使用 stratified split；若條件不足則退而求其次用一般切分
    if len(rest_df) >= 10 and (not rest_df[strat_col].isna().any()):
        vc2 = rest_df[strat_col].value_counts()
        if len(vc2) > 1 and vc2.min() >= 2:
            train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed, stratify=rest_df[strat_col])
        else:
            train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed)
    elif len(rest_df) >= 2:
        train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=seed)
    else:
        train_df = rest_df.copy()
        val_df = rest_df.iloc[:0].copy()

    # 再把被保護的 rare 類別加回 train
    train_df = pd.concat([train_df, forced_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df, rare_summary


def score_split_class_coverage(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_cols: List[str],
) -> Tuple[int, int, int, float]:
    """Evaluate how well a split covers classes across all labels."""
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

            # Train missing is the most serious case: the model cannot learn that class.
            if train_count == 0:
                train_missing += 1

            # A class with only one sample cannot appear in both train and validation.
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
    """Create one candidate split using a selected stratify column when possible."""
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
    """Try multiple candidate seeds and keep the split with best class coverage."""
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

        # 修改重點：不只用單一 random_state，改為多試幾個候選切分。
        # 評分時會同時看所有 label，挑 validation 缺失類別最少且 train 不缺類別的 split。
        for candidate_strat_col in [strat_col, None] + [lbl for lbl in label_cols if lbl != strat_col]:
            train_df, val_df = split_with_strategy(rest_df, forced_df, candidate_strat_col, candidate_seed, group_col=group_col)
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


def split_sample_key(df: pd.DataFrame) -> pd.Series:
    """Build a stable ear-level key for optional split manifests."""
    if "case_id" not in df.columns or "ear_side" not in df.columns:
        raise KeyError("split manifest requires case_id and ear_side columns in task dataframe")
    return df["case_id"].astype(str) + "__" + df["ear_side"].astype(str)


def load_split_manifest(path: str):
    if path is None or str(path).strip() == "":
        return None
    manifest = pd.read_csv(path)
    required = {"task", "case_id", "ear_side"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest missing required columns: {missing}")
    if "split" not in manifest.columns and "split_role" not in manifest.columns:
        raise ValueError("Split manifest must contain either 'split' or 'split_role'.")
    if "seed" not in manifest.columns and "fold_seed" not in manifest.columns:
        raise ValueError("Split manifest must contain either 'seed' or 'fold_seed'.")
    return manifest


def split_dataframe_from_manifest(
    df: pd.DataFrame,
    task_name: str,
    seed: int,
    split_manifest: pd.DataFrame,
):
    """Use a precomputed ear-level split manifest instead of creating a new split."""
    work = df.copy()
    seed_col = "seed" if "seed" in split_manifest.columns else "fold_seed"
    split_col = "split" if "split" in split_manifest.columns else "split_role"
    manifest = split_manifest.copy()
    manifest["_seed_str"] = manifest[seed_col].astype(str)
    sub = manifest[
        (manifest["task"].astype(str) == str(task_name))
        & (manifest["_seed_str"] == str(seed))
    ].copy()
    if len(sub) == 0:
        raise ValueError(f"No split manifest rows found for task={task_name}, seed={seed}")

    if "sample_key" not in sub.columns:
        sub["sample_key"] = sub["case_id"].astype(str) + "__" + sub["ear_side"].astype(str)
    sub["_split_norm"] = sub[split_col].astype(str).str.strip().str.lower()
    train_keys = set(sub.loc[sub["_split_norm"].eq("train"), "sample_key"].astype(str))
    val_keys = set(sub.loc[sub["_split_norm"].isin(["val", "valid", "validation"]), "sample_key"].astype(str))
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError(f"Split manifest train/val overlap for {task_name} seed={seed}: {len(overlap)} samples")
    if not train_keys or not val_keys:
        raise ValueError(
            f"Split manifest must provide non-empty train and val rows for {task_name} seed={seed}"
        )

    work["_sample_key"] = split_sample_key(work)
    train_df = work.loc[work["_sample_key"].isin(train_keys)].drop(columns=["_sample_key"]).reset_index(drop=True)
    val_df = work.loc[work["_sample_key"].isin(val_keys)].drop(columns=["_sample_key"]).reset_index(drop=True)
    test_keys = set(sub.loc[sub["_split_norm"].eq("test"), "sample_key"].astype(str))
    test_df = work.loc[work["_sample_key"].isin(test_keys)].drop(columns=["_sample_key"]).reset_index(drop=True)
    assigned_keys = train_keys | val_keys | test_keys
    unassigned_count = int((~work["_sample_key"].isin(assigned_keys)).sum())
    rare_summary = {
        "_split_manifest_enabled": True,
        "_split_manifest_seed": int(seed),
        "_split_manifest_rows": int(len(sub)),
        "_split_manifest_train_rows": int(len(train_df)),
        "_split_manifest_val_rows": int(len(val_df)),
        "_split_manifest_test_rows": int(len(test_df)),
        "_split_manifest_unassigned_rows": unassigned_count,
        "_group_split_enabled": True,
        "_group_overlap_count": int(
            len(set(train_df["case_id"].astype(str)) & set(val_df["case_id"].astype(str)))
        ) if "case_id" in train_df.columns and "case_id" in val_df.columns else None,
    }
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Split manifest produced empty train/val for {task_name} seed={seed}")
    return train_df, val_df, test_df, rare_summary


def compute_norm_meta(train_df: pd.DataFrame, feature_cols: List[str]):
    """從 train set 計算標準化所需的平均與標準差。"""
    mu = {}
    sigma = {}
    for c in feature_cols:
        m = float(train_df[c].mean())
        s = float(train_df[c].std(ddof=0))
        if not np.isfinite(s) or s < 1e-8:
            s = 1.0
        mu[c] = m
        sigma[c] = s
    return {"mu": mu, "sigma": sigma}


def apply_norm(df: pd.DataFrame, feature_cols: List[str], mu: Dict[str, float], sigma: Dict[str, float]):
    """依指定 mu / sigma 對資料表做 z-score 標準化。"""
    out = df.copy()
    for c in feature_cols:
        out[c] = (out[c].astype(float) - mu[c]) / sigma[c]
    return out


def normalized_raw_scalar(feature: str, raw_value: float, norm_for_task: Dict[str, Dict[str, float]]) -> float:
    """Convert a raw feature value into the task-specific normalized tensor scale."""
    mu = float(norm_for_task.get("mu", {}).get(feature, 0.0))
    sigma = float(norm_for_task.get("sigma", {}).get(feature, 1.0))
    if not np.isfinite(sigma) or abs(sigma) < 1e-8:
        sigma = 1.0
    return float((float(raw_value) - mu) / sigma)


def set_augmented_feature(
    X: torch.Tensor,
    row_mask: torch.Tensor,
    feature_to_idx: Dict[str, int],
    norm_for_task: Dict[str, Dict[str, float]],
    feature: str,
    raw_value: float,
) -> bool:
    idx = feature_to_idx.get(feature)
    if idx is None:
        return False
    X[row_mask, idx] = normalized_raw_scalar(feature, raw_value, norm_for_task)
    return True


def effective_missing_aug_strategy_weights(configured: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    weights = copy.deepcopy(DEFAULT_MISSING_AUG_STRATEGY_WEIGHTS)
    if isinstance(configured, dict):
        for task_name, task_weights in configured.items():
            if task_name not in weights or not isinstance(task_weights, dict):
                continue
            for strategy_name, value in task_weights.items():
                try:
                    weights[task_name][strategy_name] = max(0.0, float(value))
                except (TypeError, ValueError):
                    continue
    return weights


def sample_missingness_strategy(
    task_name: str,
    strategy_names: List[str],
    n_rows: int,
    device: torch.device,
    strategy_weights: Dict[str, Dict[str, float]] = None,
) -> torch.Tensor:
    configured = effective_missing_aug_strategy_weights(strategy_weights).get(task_name, {})
    weights = torch.tensor(
        [max(0.0, float(configured.get(name, 0.0))) for name in strategy_names],
        dtype=torch.float32,
        device=device,
    )
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones(len(strategy_names), dtype=torch.float32, device=device)
    probabilities = weights / weights.sum()
    return torch.multinomial(probabilities, n_rows, replacement=True)


def apply_training_missingness_augmentation(
    X: torch.Tensor,
    task_name: str,
    feature_cols: List[str],
    norm_for_task: Dict[str, Dict[str, float]],
    missing_aug_p: float = 0.0,
    missing_aug_strategy_weights: Dict[str, Dict[str, float]] = None,
) -> torch.Tensor:
    """Apply training-only structured missingness perturbations on normalized tensors."""
    p = max(0.0, min(1.0, float(missing_aug_p or 0.0)))
    if p <= 0.0 or X.numel() == 0 or task_name not in {"Task1", "Task2", "Task3"}:
        return X

    selected = torch.rand(X.shape[0], device=X.device) < p
    if not bool(selected.any().item()):
        return X

    out = X.clone()
    feature_to_idx = {feature: idx for idx, feature in enumerate(feature_cols)}

    if task_name == "Task1":
        strategy_names = ["mask_pta", "mask_high_freq", "mask_low_freq", "mask_all_ac"]
        strategy = sample_missingness_strategy(
            task_name,
            strategy_names,
            X.shape[0],
            X.device,
            missing_aug_strategy_weights,
        )
        masks = {name: selected & (strategy == idx) for idx, name in enumerate(strategy_names)}

        def mask_task1_features(row_mask: torch.Tensor, features: List[str]) -> None:
            if not bool(row_mask.any().item()):
                return
            for feature in features:
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, feature, 0.0)

        mask_task1_features(masks["mask_pta"], ["ac_PTA"])
        mask_task1_features(masks["mask_high_freq"], ["ac_2000Hz", "ac_4000Hz", "ac_PTA"])
        mask_task1_features(masks["mask_low_freq"], ["ac_500Hz", "ac_1000Hz", "ac_PTA"])
        mask_task1_features(masks["mask_all_ac"], TASK1_EAR_LEVEL_FEATURES)

    if task_name == "Task2":
        strategy_names = ["no_bc", "partial_bc", "multi_ac_nr"]
        strategy = sample_missingness_strategy(
            task_name,
            strategy_names,
            X.shape[0],
            X.device,
            missing_aug_strategy_weights,
        )
        masks = {name: selected & (strategy == idx) for idx, name in enumerate(strategy_names)}

        for hz in TASK2_ABG_FREQS:
            for mask_name in ("no_bc", "partial_bc"):
                row_mask = masks[mask_name]
                if mask_name == "partial_bc" and hz not in (500, 1000):
                    continue
                if not bool(row_mask.any().item()):
                    continue
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"bc_{hz}Hz", 0.0)
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"bc_{hz}Hz_nr", 0.0)
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"bc_{hz}Hz_missing", 1.0)
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"abg_{hz}Hz", 0.0)
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"abg_{hz}Hz_missing", 1.0)
                set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"abg_{hz}Hz_censored", 0.0)

        nr_mask = masks["multi_ac_nr"]
        if bool(nr_mask.any().item()):
            for hz in (2000, 4000, 6000, 8000):
                set_augmented_feature(
                    out,
                    nr_mask,
                    feature_to_idx,
                    norm_for_task,
                    f"ac_{hz}Hz",
                    TASK2_AC_NR_LIMITS_DB.get(hz, 120.0),
                )
                set_augmented_feature(out, nr_mask, feature_to_idx, norm_for_task, f"ac_{hz}Hz_nr", 1.0)
                if hz in TASK2_ABG_FREQS:
                    set_augmented_feature(out, nr_mask, feature_to_idx, norm_for_task, f"abg_{hz}Hz_censored", 1.0)
                    set_augmented_feature(out, nr_mask, feature_to_idx, norm_for_task, f"abg_{hz}Hz_missing", 0.0)

    if task_name == "Task3":
        strategy_names = ["no_peak", "no_width", "np_like", "no_tymp"]
        strategy = sample_missingness_strategy(
            task_name,
            strategy_names,
            X.shape[0],
            X.device,
            missing_aug_strategy_weights,
        )
        masks = {name: selected & (strategy == idx) for idx, name in enumerate(strategy_names)}

        def mark_missing(row_mask: torch.Tensor, col: str) -> None:
            if not bool(row_mask.any().item()):
                return
            set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, col, 0.0)
            set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"{col}_real_zero", 0.0)
            set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"{col}_missing_zero", 1.0)
            set_augmented_feature(out, row_mask, feature_to_idx, norm_for_task, f"{col}_np_zero", 0.0)

        for col in ("tymp_peak_daPa",):
            mark_missing(masks["no_peak"], col)
        for col in ("tymp_Width_daPa",):
            mark_missing(masks["no_width"], col)
        for col in TASK3_EAR_LEVEL_BASE_FEATURES:
            mark_missing(masks["no_tymp"], col)

        np_mask = masks["np_like"]
        if bool(np_mask.any().item()):
            np_values = {
                "tymp_peak_daPa": -999.0,
                "tymp_peak_mmho": -1.0,
                "tymp_Vea": 0.0,
                "tymp_Width_daPa": 0.0,
            }
            for col, raw_value in np_values.items():
                set_augmented_feature(out, np_mask, feature_to_idx, norm_for_task, col, raw_value)
                set_augmented_feature(out, np_mask, feature_to_idx, norm_for_task, f"{col}_real_zero", 0.0)
                set_augmented_feature(out, np_mask, feature_to_idx, norm_for_task, f"{col}_missing_zero", 0.0)
                set_augmented_feature(out, np_mask, feature_to_idx, norm_for_task, f"{col}_np_zero", 1.0)

    return out


def build_class_weight_tensor(y_series: pd.Series, n_classes: int):
    """根據各類別樣本數建立 class weight，平衡不均衡資料。"""
    counts = y_series.value_counts().to_dict()
    total = len(y_series)
    weights = []
    for i in range(n_classes):
        c = counts.get(i, 0)
        if c <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (n_classes * c))
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    return w


def build_multilabel_sample_weights(
    df: pd.DataFrame,
    label_cols: List[str],
    max_weight: float = None,
):
    """根據多個 label 的類別頻率建立 sample weight，供 Task2 sampler 使用。"""
    if len(df) == 0:
        return torch.ones(0, dtype=torch.double)

    per_label = []
    for lbl in label_cols:
        counts = df[lbl].value_counts().to_dict()
        n_classes = max(len(counts), 1)
        total = len(df)
        weights = df[lbl].map(lambda x: total / (n_classes * max(counts.get(x, 0), 1))).astype(float)
        per_label.append(weights)

    sample_weights = pd.concat(per_label, axis=1).mean(axis=1)
    if max_weight is not None:
        sample_weights = sample_weights.clip(upper=float(max_weight))
    return torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double)


############################################################################################################

def norm_pred_name(pred_name: str) -> str:
    """把預測類別名稱標準化，避免 MODERATE 誤匹配到 MODERATELY。"""
    return str(pred_name).strip().upper()


def infer_task2_hearing_type_from_rule(row: dict) -> Tuple[str, bool]:
    return shared_infer_task2_hearing_type_from_rule(row)


def expert_consistency_score(task_name: str, label_col: str, row: dict, pred_name: str):
    """以簡化臨床規則評估預測結果是否合理。

    建議分數：
    - 強符合：1.0
    - 弱符合 / borderline：0.6
    - 不符合：-0.2
    """
    return float(score_rule_decision(task_name, row, pred_name))


def expert_consistency_target_and_confidence(
    task_name: str,
    label_col: str,
    row: dict,
    pred_name: str,
):
    """Return reward target and confidence weight for the expert rule signal."""
    target = float(expert_consistency_score(task_name, label_col, row, pred_name))
    decision = rule_decision(task_name, row)
    confidence = float(decision.confidence) if decision.covered else DEFAULT_REWARD_CONFIDENCE_FLOOR
    confidence = max(float(confidence), DEFAULT_REWARD_CONFIDENCE_FLOOR)
    return target, confidence

############################################################################################################


def build_episode(
    X: torch.Tensor,
    ys: List[torch.Tensor],
    label_cols: List[str],
    max_support_per_class: int = 2,
):
    """Split one batch into disjoint support/query parts for prototype meta mode.

    NOTE 2026-05-20 P2-12:
    問題：舊流程從 batch 取 support 後，仍用同一批 X 算 loss，support/query 沒有分開。
    使用步驟：training loop 和 evaluate_task() 在 enable_meta=True 時呼叫本函式。
    功能例子：某 batch 中 SNHL 有 5 筆、WNL 有 3 筆且 support_per_class=2，
    則 SNHL/WNL 各取最多 2 筆做 support，其餘樣本作 query 來計算 CE/reward。
    """
    if X.size(0) == 0 or not ys:
        query_idx = torch.arange(X.size(0), dtype=torch.long, device=X.device)
        return None, query_idx

    max_support_per_class = max(int(max_support_per_class), 0)
    y_ref = ys[0]
    y_cpu = y_ref.detach().cpu().numpy()

    support_indices = []
    for class_idx in sorted(set(y_cpu.tolist())):
        class_indices = np.where(y_cpu == class_idx)[0]
        if len(class_indices) <= 1 or max_support_per_class == 0:
            continue
        n_support = min(max_support_per_class, len(class_indices) - 1)
        support_indices.extend(class_indices[:n_support].tolist())

    support_indices = sorted(set(support_indices))
    support_set = set(support_indices)
    query_indices = [idx for idx in range(len(y_cpu)) if idx not in support_set]
    if not query_indices:
        query_indices = list(range(len(y_cpu)))
        support_indices = []

    query_idx = torch.tensor(query_indices, dtype=torch.long, device=X.device)
    if not support_indices:
        return None, query_idx

    support_idx = torch.tensor(support_indices, dtype=torch.long, device=X.device)
    support = {}
    for lbl, y in zip(label_cols, ys):
        support[lbl] = (
            X.index_select(0, support_idx),
            y.index_select(0, support_idx),
        )
    return support, query_idx


def select_query_batch(X: torch.Tensor, ys: List[torch.Tensor], raws: List[dict], query_idx: torch.Tensor):
    """Select query samples while keeping raw rows aligned with tensors.

    NOTE 2026-05-20 P2-12:
    問題：query 只選 tensor 不同步 raw rows，reward rule 可能拿錯原始資料列。
    使用步驟：build_episode() 回傳 query_idx 後，training/evaluate 會用本函式切 X、y、raws。
    功能例子：query_idx=[2, 4] 時，X_query、ys_query 和 raws_query 都只保留第 2、4 筆，expert_consistency_score() 才會對到正確病患資料。
    """
    if query_idx is None:
        return X, ys, raws
    query_list = [int(i) for i in query_idx.detach().cpu().tolist()]
    if len(query_list) == X.size(0):
        return X, ys, raws
    X_query = X.index_select(0, query_idx)
    ys_query = [y.index_select(0, query_idx) for y in ys]
    raws_query = [raws[i] for i in query_list]
    return X_query, ys_query, raws_query


def safe_multiclass_auc(y_true, prob, n_classes: int):
    """安全計算多分類 macro AUC。

    當類別數不足或資料不合法時回傳 None。
    """
    try:
        y_true = np.asarray(y_true, dtype=int).ravel()
        prob = np.asarray(prob, dtype=float)
        if len(y_true) == 0 or prob.ndim != 2 or prob.shape[1] < n_classes:
            return None

        aucs = []
        for class_idx in range(n_classes):
            # 修改重點：validation 若沒有此類別，這個 one-vs-rest AUC 在數學上不可定義。
            # 因此只計算同時具有正樣本與負樣本的類別，避免 sklearn UndefinedMetricWarning。
            y_binary = (y_true == class_idx).astype(int)
            if len(np.unique(y_binary)) < 2:
                continue
            aucs.append(float(roc_auc_score(y_binary, prob[:, class_idx])))

        if not aucs:
            return None
        return float(np.mean(aucs))
    except Exception:
        return None


def safe_balanced_accuracy(y_true, y_pred, n_classes: int):
    """安全計算 balanced accuracy，避免 y_pred 出現 validation 缺席類別時洗版。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if len(y_true) == 0:
        return 0.0

    recalls = []
    for class_idx in range(n_classes):
        support = int(np.sum(y_true == class_idx))
        if support == 0:
            continue

        # 修改重點：只平均 validation 真實答案中存在的類別。
        # 若模型預測到 y_true 沒有的類別，該筆仍會算錯，但不再觸發 sklearn warning。
        correct = int(np.sum((y_true == class_idx) & (y_pred == class_idx)))
        recalls.append(correct / support)

    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def safe_class_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")


def per_class_report(y_true, y_pred, class_names: List[str], label_indices: List[int]) -> Dict[str, Dict[str, float]]:
    """Return per-class precision/recall/F1/support with stable class names.

    NOTE 2026-05-20 P2-15:
    問題：只看 macro-F1 會掩蓋少數類別，例如 Task2 的 CHL 或 Task3 的 B/C。
    使用步驟：evaluate_task() 每個 label 都會呼叫本函式，產生各 class 的 precision/recall/F1/support。
    功能例子：Task2 整體 macro-F1 上升，但 CHL recall=0 時，per-class report 會直接顯示 CHL 仍未學好。
    """
    if not y_true:
        return {
            class_name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for class_name in class_names
        }

    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    out = {}
    for class_name in class_names:
        row = report.get(class_name, {})
        out[class_name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }
    return out


def evaluate_task(
    model,
    loader,
    meta,
    task_name,
    criteria_dict,
    use_meta_support: bool = False,
    support_per_class: int = 2,
    return_predictions: bool = False,
):
    """評估單一 task，輸出 metrics 與 confusion matrix。

    NOTE 2026-05-20 P2-12/P2-15:
    P2-12：use_meta_support=True 時，validation 也會切 support/query，讓 meta/prototype 效果真的被評估。
    P2-15：評估除了 acc/macro-F1/balanced_acc/AUC，也會輸出每類 precision/recall/F1/support。
    功能例子：full 實驗會用 support_per_class 建 validation support；no_meta 則 use_meta_support=False，維持一般 supervised evaluation。
    """
    model.eval()
    label_cols = meta["tasks"][task_name]["label_cols"]
    class_names = meta["tasks"][task_name]["class_names"]

    all_true = {lbl: [] for lbl in label_cols}
    all_pred = {lbl: [] for lbl in label_cols}
    all_prob = {lbl: [] for lbl in label_cols}
    prediction_rows = []

    total_loss = 0.0
    with torch.no_grad():
        for X, ys, raws in loader:
            X = X.to(DEVICE)
            ys = [y.to(DEVICE) for y in ys]
            support = None
            X_eval, ys_eval, raws_eval = X, ys, raws
            if use_meta_support:
                support, query_idx = build_episode(
                    X,
                    ys,
                    label_cols,
                    max_support_per_class=support_per_class,
                )
                X_eval, ys_eval, raws_eval = select_query_batch(X, ys, raws, query_idx)

            logits_dict, reward_dict = model(X_eval, task_name, support=support)
            batch_loss = 0.0
            for lbl, y in zip(label_cols, ys_eval):
                logits = logits_dict[lbl]
                loss = criteria_dict[lbl](logits, y)
                batch_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_true[lbl].extend(y.cpu().tolist())
                all_pred[lbl].extend(preds.cpu().tolist())
                all_prob[lbl].extend(probs.cpu().tolist())
                if return_predictions:
                    probs_np = probs.detach().cpu().numpy()
                    preds_np = preds.detach().cpu().numpy()
                    true_np = y.detach().cpu().numpy()
                    rewards = reward_dict.get(lbl)
                    reward_np = (
                        rewards.detach().cpu().numpy()
                        if rewards is not None
                        else np.full(len(preds_np), np.nan, dtype=float)
                    )
                    names = class_names[lbl]
                    feature_cols = TASK_INFO.get(task_name, {}).get("feature_cols", [])
                    for i, raw in enumerate(raws_eval):
                        true_idx = int(true_np[i])
                        pred_idx = int(preds_np[i])
                        true_name = names[true_idx] if 0 <= true_idx < len(names) else str(true_idx)
                        pred_name = names[pred_idx] if 0 <= pred_idx < len(names) else str(pred_idx)
                        expert_target, expert_confidence = expert_consistency_target_and_confidence(
                            task_name,
                            lbl,
                            raw,
                            pred_name,
                        )
                        row = {
                            "task": task_name,
                            "label_col": lbl,
                            "case_id": raw.get("case_id"),
                            "ear_side": raw.get("ear_side"),
                            "true_idx": true_idx,
                            "pred_idx": pred_idx,
                            "true_label": true_name,
                            "pred_label": pred_name,
                            "correct": bool(true_idx == pred_idx),
                            "confidence": float(probs_np[i, pred_idx]) if probs_np.shape[1] > pred_idx else np.nan,
                            "reward_pred": float(reward_np[i]) if i < len(reward_np) else np.nan,
                            "expert_target": float(expert_target),
                            "expert_confidence": float(expert_confidence),
                            "prob_json": json.dumps(
                                {str(names[j]): float(probs_np[i, j]) for j in range(len(names))},
                                ensure_ascii=False,
                            ),
                        }
                        for feature_col in feature_cols:
                            row[f"feature__{feature_col}"] = raw.get(feature_col)
                        prediction_rows.append(row)
            total_loss += batch_loss

    metrics = {"loss": total_loss / max(len(loader), 1)}
    cm_store = {}
    for lbl in label_cols:
        y_true = all_true[lbl]
        y_pred = all_pred[lbl]
        probs = np.array(all_prob[lbl], dtype=float) if len(all_prob[lbl]) else np.zeros((0, len(class_names[lbl])))
        n_classes = len(class_names[lbl])

        # 常見分類指標
        metrics[f"{lbl}_acc"] = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        metrics[f"{lbl}_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
        metrics[f"{lbl}_balanced_acc"] = safe_balanced_accuracy(y_true, y_pred, n_classes) if y_true else 0.0
        auc = safe_multiclass_auc(y_true, probs, n_classes) if len(y_true) else None
        metrics[f"{lbl}_macro_auc_ovr"] = auc

        # confusion matrix
        labels = list(range(n_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels) if y_true else np.zeros((len(labels), len(labels)), dtype=int)
        cm_store[lbl] = cm.tolist()

        class_report = per_class_report(y_true, y_pred, class_names[lbl], labels)
        metrics[f"{lbl}_per_class"] = class_report
        for class_name, class_metrics in class_report.items():
            safe_name = safe_class_name(class_name)
            for metric_name, metric_value in class_metrics.items():
                metrics[f"{lbl}__class__{safe_name}__{metric_name}"] = metric_value
    if return_predictions:
        return metrics, cm_store, pd.DataFrame(prediction_rows)
    return metrics, cm_store


def save_json(obj, path: str):
    """將 Python 物件存成格式化 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_run_configs_csv(configs: List[Dict], out_path: Path):
    """輸出超參數紀錄 CSV，方便對照每次訓練設定。"""
    rows = []
    for idx, cfg in enumerate(configs, start=1):
        row = {"config_index": idx}
        row.update(cfg)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_per_class_metrics_csv(eval_summary: Dict[str, Dict[str, object]], out_path: Path):
    """Write nested per-class metrics into a flat CSV table.

    NOTE 2026-05-20 P2-15:
    問題：eval_summary.json 的巢狀 per-class 結構適合程式讀，但不方便人工比較多個 run。
    使用步驟：run_experiment() final evaluation 後呼叫，輸出 run_dir/per_class_metrics.csv。
    功能例子：可直接用表格篩選 task=Task2、class_name=CHL，比較 run_01 與 run_09_tiny 的 recall。
    """
    rows = []
    for task_name, metrics in eval_summary.items():
        for metric_key, report in metrics.items():
            if not metric_key.endswith("_per_class") or not isinstance(report, dict):
                continue
            label_col = metric_key[:-len("_per_class")]
            for class_name, class_metrics in report.items():
                row = {
                    "task": task_name,
                    "label_col": label_col,
                    "class_name": class_name,
                }
                row.update(class_metrics)
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def reset_runs_root(runs_root: Path):
    """清空 five_runs 輸出資料夾，避免混入前一次結果。"""
    if runs_root.exists():
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)


def get_experiment_config(
    exp_name: str,
    base_reward_weight: float,
    single_task_target: str = "Task2",
) -> Tuple[float, bool, List[str]]:
    """集中管理各實驗模式的 reward、meta-learning 與訓練任務清單。"""
    reward_weight = base_reward_weight
    enable_meta = True
    task_sequence = ["Task1", "Task2", "Task3"]
    if exp_name in SINGLE_TASK_EXPERIMENT_ALIASES:
        single_task_target = SINGLE_TASK_EXPERIMENT_ALIASES[exp_name]
        exp_name = "single_task"

    if exp_name == "no_irl":
        reward_weight = 0.0
    elif exp_name == "no_meta":
        enable_meta = False
    elif exp_name == "single_task":
        if single_task_target not in SINGLE_TASK_TARGETS:
            raise ValueError(f"single_task_target must be one of {SINGLE_TASK_TARGETS}: {single_task_target}")
        task_sequence = [single_task_target]
    elif exp_name != "full":
        raise ValueError(f"Unknown experiment: {exp_name}")

    return reward_weight, enable_meta, task_sequence


def expand_experiment_names(experiments: List[str], single_task_target: str) -> List[str]:
    """Expand single_task into explicit Task1/Task2/Task3 aliases when requested."""
    expanded = []
    for exp_name in experiments:
        if exp_name == "single_task":
            if single_task_target == "all":
                expanded.extend(["single_task_task1", "single_task_task2", "single_task_task3"])
            elif single_task_target == "Task2":
                expanded.append("single_task")
            elif single_task_target in SINGLE_TASK_TARGETS:
                expanded.append(f"single_task_{single_task_target.lower()}")
            else:
                raise ValueError(
                    f"single_task_target must be Task1, Task2, Task3, or all: {single_task_target}"
                )
        else:
            expanded.append(exp_name)
    return expanded


def parse_optional_int(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if str(value).strip() == "":
        return None
    return int(value)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_task_step_targets(loaders_train: Dict[str, DataLoader], task_sequence: List[str], mode: str, steps_per_task: int = None):
    """Decide how many optimizer steps each task gets in one epoch.

    NOTE 2026-05-20 P2-14:
    問題：三個 task 資料量差很多，完整跑 loader 會讓大資料 task 拿到更多更新步數。
    使用步驟：run_experiment() 建好 loaders 後呼叫本函式，training loop 再依回傳值抽 batch。
    功能例子：mode=equal_steps 且 steps_per_task=None 時，會使用最大 loader 長度作為每個 task 的共同步數。
    """
    if mode == "full_loader":
        return {task_name: len(loaders_train[task_name]) for task_name in task_sequence}
    if mode == "equal_steps":
        target = steps_per_task if steps_per_task is not None else max(len(loaders_train[task_name]) for task_name in task_sequence)
        return {task_name: int(target) for task_name in task_sequence}
    if mode.startswith("round_robin_k"):
        target = steps_per_task if steps_per_task is not None else max(len(loaders_train[task_name]) for task_name in task_sequence)
        return {task_name: int(target) for task_name in task_sequence}
    raise ValueError(f"Unknown task_sampling_mode: {mode}")


def parse_round_robin_k(mode: str) -> int:
    """Extract k from task_sampling_mode like round_robin_k2."""
    prefix = "round_robin_k"
    if not mode.startswith(prefix):
        return 0
    try:
        return max(int(mode[len(prefix):]), 1)
    except ValueError as exc:
        raise ValueError(f"Invalid round-robin task_sampling_mode: {mode}") from exc


def build_task_step_schedule(
    task_sequence: List[str],
    effective_steps_per_task: Dict[str, int],
    task_sampling_mode: str,
) -> List[str]:
    """Create the task order used inside one epoch."""
    if task_sampling_mode in {"full_loader", "equal_steps"}:
        schedule = []
        for task_name in task_sequence:
            schedule.extend([task_name] * int(effective_steps_per_task[task_name]))
        return schedule

    k = parse_round_robin_k(task_sampling_mode)
    if k <= 0:
        raise ValueError(f"Unknown task_sampling_mode: {task_sampling_mode}")

    remaining = {task_name: int(effective_steps_per_task[task_name]) for task_name in task_sequence}
    schedule = []
    while any(remaining[task_name] > 0 for task_name in task_sequence):
        for task_name in task_sequence:
            take = min(k, remaining[task_name])
            if take <= 0:
                continue
            schedule.extend([task_name] * take)
            remaining[task_name] -= take
    return schedule


def next_cycled_batch(loader, iterator):
    """Get the next batch and restart the loader iterator when exhausted.

    NOTE 2026-05-20 P2-14:
    問題：equal_steps 可能要求 Task2/Task3 跑超過自身 loader 長度，需要安全循環抽 batch。
    使用步驟：training loop 每一步都用本函式取 batch，而不是直接 for X, ys, raws in loader。
    功能例子：Task2 loader 只有 1 個 batch，但 target_steps=3 時，會抽完後重新 iter(loader) 繼續抽。
    """
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def run_experiment(
    data_dir: str,
    out_root: str,
    seed: int,
    exp_name: str,
    run_cfg: Dict[str, float],
    split_manifest: pd.DataFrame = None,
    single_task_target: str = "Task2",
):
    """執行單次實驗（某個 exp_name + 某個 seed）。"""
    seed_everything(seed)
    lr = float(run_cfg["lr"])
    batch_size = int(run_cfg["batch"])
    epochs = int(run_cfg["epochs"])
    patience = int(run_cfg["patience"])
    weight_decay = float(run_cfg["weight_decay"])
    base_reward_weight = float(run_cfg["reward_weight"])
    model_config = {
        "d_model": int(run_cfg.get("d_model", DEFAULT_MODEL_CONFIG["d_model"])),
        "nhead": int(run_cfg.get("nhead", DEFAULT_MODEL_CONFIG["nhead"])),
        "num_layers": int(run_cfg.get("num_layers", DEFAULT_MODEL_CONFIG["num_layers"])),
        "dropout": float(run_cfg.get("dropout", DEFAULT_MODEL_CONFIG["dropout"])),
        "proto_alpha": float(run_cfg.get("proto_alpha", DEFAULT_MODEL_CONFIG["proto_alpha"])),
        "proto_temperature": float(run_cfg.get("proto_temperature", DEFAULT_MODEL_CONFIG["proto_temperature"])),
    }
    if model_config["d_model"] % model_config["nhead"] != 0:
        raise ValueError(f"d_model must be divisible by nhead: {model_config}")

    # NOTE 2026-05-20 P2-12/P2-14:
    # 這裡集中讀取 run_cfg 中的 meta episode 與 task sampling 設定。
    # P2-12 使用 support_per_class/validate_with_meta_support 控制 support/query；
    # P2-14 使用 task_sampling_mode/steps_per_task 控制每個 task 的訓練步數。
    task_sampling_mode = str(run_cfg.get("task_sampling_mode", DEFAULT_TASK_SAMPLING_CONFIG["task_sampling_mode"]))
    steps_per_task = parse_optional_int(run_cfg.get("steps_per_task", DEFAULT_TASK_SAMPLING_CONFIG["steps_per_task"]))
    support_per_class = parse_optional_int(run_cfg.get("support_per_class", DEFAULT_META_EPISODE_CONFIG["support_per_class"]))
    if support_per_class is None:
        support_per_class = DEFAULT_META_EPISODE_CONFIG["support_per_class"]
    support_per_class = max(int(support_per_class), 0)
    validate_with_meta_support = parse_bool(run_cfg.get(
        "validate_with_meta_support",
        DEFAULT_META_EPISODE_CONFIG["validate_with_meta_support"],
    ))
    missing_aug_p = max(0.0, min(1.0, float(run_cfg.get("missing_aug_p", 0.0) or 0.0)))
    missing_aug_strategy_weights = effective_missing_aug_strategy_weights(
        run_cfg.get("missing_aug_strategy_weights", {})
    )

    reward_weight, enable_meta, task_sequence = get_experiment_config(
        exp_name,
        base_reward_weight,
        single_task_target=single_task_target,
    )

    run_dir = Path(out_root) / f"{exp_name}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MetaIRL v7.4] Using device: {DEVICE}")
    print(f"[MetaIRL v7.4] Run dir: {run_dir}")
    print(f"[MetaIRL v7.4] Model config: {model_config}")
    print(f"[MetaIRL v7.4] Task sampling: mode={task_sampling_mode}, steps_per_task={steps_per_task}")
    print(f"[MetaIRL v7.4] Meta episode: support_per_class={support_per_class}, validate_with_meta_support={validate_with_meta_support}")
    print(f"[MetaIRL v7.4] Training missingness augmentation p={missing_aug_p:.3f}")
    if missing_aug_p > 0.0:
        print(f"[MetaIRL v7.4] Missingness augmentation weights: {missing_aug_strategy_weights}")
    print(f"[MetaIRL v7.4] Loading CSVs from: {Path(data_dir).resolve()}")

    # ===== 讀取所有 task 的原始 CSV，並建立 union_features =====
    raw_dfs = {}
    union_features = set()
    for task_name, info in TASK_INFO.items():
        csv_path = Path(data_dir) / info["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {task_name}: {csv_path.resolve()}")
        df = load_tabular_data(csv_path)
        if task_name == "Task1":
            df = prepare_task1_dataframe(df)
        if task_name == "Task2":
            # 修改：只在記憶體中補出 Task2 ABG/missing-mask 特徵，不改原始 CSV。
            df = prepare_task2_dataframe(df)
        if task_name == "Task3":
            df = prepare_task3_dataframe(df)
        raw_dfs[task_name] = df
        union_features.update(info["feature_cols"])
    union_features = sorted(list(union_features))
    print(f"[MetaIRL v7.4] Union features = {len(union_features)}")

    # 用來存每個 task 的資料、標籤資訊、DataLoader、loss function 等
    tasks_meta = {}
    task_label_class_counts = {}
    loaders_train = {}
    loaders_val = {}
    loaders_test = {}
    criteria_by_task = {}
    norm_meta = {}
    train_sizes = {}
    test_sizes = {}
    cleaning_log = {}

    # ===== 逐一處理三個 task 的資料 =====
    for task_name, info in TASK_INFO.items():
        df = pad_and_clean(raw_dfs[task_name], union_features)
        df, missing_stats = clean_label_columns(df, info["label_cols"])
        cleaning_log[task_name] = {"missing_label_stats": missing_stats}
        if raw_dfs[task_name].attrs:
            cleaning_log[task_name]["preprocess_log"] = dict(raw_dfs[task_name].attrs)
        if len(df) < 2:
            raise ValueError(f"{task_name} has too few rows after dropping missing labels: {len(df)}")

        if split_manifest is not None:
            train_df, val_df, test_df, rare_summary = split_dataframe_from_manifest(df, task_name, seed, split_manifest)
        else:
            train_df, val_df, rare_summary = split_dataframe(df, task_name, seed)
            test_df = df.iloc[:0].copy()
        cleaning_log[task_name]["rare_classes_forced_to_train"] = rare_summary
        support_report = build_class_support_report(df, train_df, val_df, info["label_cols"])
        cleaning_log[task_name]["class_support"] = support_report
        for split_name, split_report in support_report.items():
            for lbl, lbl_report in split_report.items():
                low_support = lbl_report.get("low_support_classes", {})
                if low_support:
                    print(
                        f"[WARN] {task_name} {split_name} {lbl} "
                        f"low support (<{LOW_SUPPORT_THRESHOLD}): {low_support}"
                    )

        # ===== 對每個 label 建立 LabelEncoder 與 loss =====
        label_class_names = {}
        label_num_classes = {}
        criteria_dict = {}
        for lbl in info["label_cols"]:
            le = LabelEncoder()
            le.fit(df[lbl].astype(str))
            train_df.loc[:, lbl] = le.transform(train_df[lbl].astype(str))
            val_df.loc[:, lbl] = le.transform(val_df[lbl].astype(str))
            if len(test_df):
                test_df.loc[:, lbl] = le.transform(test_df[lbl].astype(str))
            label_class_names[lbl] = list(le.classes_)
            label_num_classes[lbl] = len(le.classes_)

            # 類別不均衡時使用 class-weighted cross entropy
            class_weights = build_class_weight_tensor(train_df[lbl], label_num_classes[lbl])
            criteria_dict[lbl] = nn.CrossEntropyLoss(weight=class_weights)

        task_label_class_counts[task_name] = label_num_classes
        tasks_meta[task_name] = {
            "label_cols": info["label_cols"],
            "class_names": label_class_names,
            "num_classes": label_num_classes,
        }
        criteria_by_task[task_name] = criteria_dict


        '''
        # ===== 標準化 =====
        nm = compute_norm_meta(train_df, union_features)
        norm_meta[task_name] = nm
        train_df_n = apply_norm(train_df, union_features, nm["mu"], nm["sigma"])
        val_df_n = apply_norm(val_df, union_features, nm["mu"], nm["sigma"])

        # ===== 建立 Dataset / DataLoader =====
        ds_train = MultiLabelTabularDataset(train_df_n, union_features, info["label_cols"])
        ds_val = MultiLabelTabularDataset(val_df_n, union_features, info["label_cols"])
        '''

        #################################################################修改
        # ===== 標準化 =====
        nm = compute_norm_meta(train_df, union_features)
        norm_meta[task_name] = nm

        # 先保留原始尺度資料，給 expert_consistency_score 使用
        train_df_raw = train_df.copy()
        val_df_raw = val_df.copy()
        test_df_raw = test_df.copy()

        # 模型輸入仍然使用標準化資料
        train_df_n = apply_norm(train_df, union_features, nm["mu"], nm["sigma"])
        val_df_n = apply_norm(val_df, union_features, nm["mu"], nm["sigma"])
        test_df_n = apply_norm(test_df, union_features, nm["mu"], nm["sigma"]) if len(test_df) else test_df.copy()

        # ===== 建立 Dataset / DataLoader =====
        # Dataset 同時拿標準化資料與原始尺度資料：
        # - train_df_n / val_df_n 給模型當 X
        # - train_df_raw / val_df_raw 給 expert_consistency_score 當 raw
        ds_train = MultiLabelTabularDataset(
            train_df_n,
            union_features,
            info["label_cols"],
            raw_df=train_df_raw,
        )

        ds_val = MultiLabelTabularDataset(
            val_df_n,
            union_features,
            info["label_cols"],
            raw_df=val_df_raw,
        )
        ds_test = None
        if len(test_df_n):
            ds_test = MultiLabelTabularDataset(
                test_df_n,
                union_features,
                info["label_cols"],
                raw_df=test_df_raw,
            )
        #################################################################

        loaders_train[task_name] = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_with_raw,
        )
        loaders_val[task_name] = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_with_raw)
        if ds_test is not None:
            loaders_test[task_name] = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_with_raw)
        train_sizes[task_name] = len(train_df_n)
        test_sizes[task_name] = len(test_df_n)
        
        print(
            f"[MetaIRL v7.4] {task_name} train={len(train_df_n)} "
            f"val={len(val_df_n)} test={len(test_df_n)} "
            f"dropped_missing={missing_stats['rows_dropped_any_label_missing']}"
        )

    # NOTE 2026-05-20 P2-14:
    # 訓練前先決定每個 task 在單一 epoch 的實際更新步數，避免 Task1 因 loader 較長而自然取得更多 optimizer steps。
    effective_steps_per_task = build_task_step_targets(loaders_train, task_sequence, task_sampling_mode, steps_per_task)
    task_step_schedule = build_task_step_schedule(task_sequence, effective_steps_per_task, task_sampling_mode)
    print(f"[MetaIRL v7.4] Effective train steps per epoch: {effective_steps_per_task}")
    print(f"[MetaIRL v7.4] Task step schedule preview: {task_step_schedule[:24]} total_steps={len(task_step_schedule)}")
    # NOTE 2026-05-20 P2-12:
    # no_meta 實驗不使用 support；full/no_irl/single_task 則可在 validation 同步測試 support/query meta 效果。
    use_eval_meta_support = bool(enable_meta and validate_with_meta_support)

    # ===== 建立模型 =====
    model = MTLMetaIRLTransformer(
        input_dim=len(union_features),
        task_label_class_counts=task_label_class_counts,
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        proto_alpha=model_config["proto_alpha"],
        proto_temperature=model_config["proto_temperature"],
        detach_probs_for_reward=True,
        enable_meta=enable_meta,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_score = -1e9
    best_state = None
    best_val_loss = float("inf")
    bad_macro_epochs = 0
    bad_loss_epochs = 0
    min_delta = 1e-6
    history_rows = []

    print(f"[MetaIRL v7.4] Training starts... exp={exp_name} seed={seed}")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        task_step_counts = {task_name: 0 for task_name in task_sequence}
        loader_iters = {task_name: iter(loaders_train[task_name]) for task_name in task_sequence}

        # 逐 task 訓練
        for task_name in task_step_schedule:
            loader = loaders_train[task_name]
            loader_iter = loader_iters[task_name]
            label_cols = tasks_meta[task_name]["label_cols"]
            criteria_dict = criteria_by_task[task_name]
            target_steps = 1

            # NOTE 2026-05-20 P2-14:
            # 這裡不用「跑完整 loader」，而是依 target_steps 控制每個 task 的訓練次數。
            # 例：full 模式下 Task1/Task2/Task3 都會各跑 target_steps 次，降低 Task1 主導 shared encoder 的風險。
            for _ in range(target_steps):
                (X, ys, raws), loader_iter = next_cycled_batch(loader, loader_iter)
                loader_iters[task_name] = loader_iter
                X = X.to(DEVICE)
                ys = [y.to(DEVICE) for y in ys]

                # NOTE 2026-05-20 P2-12:
                # enable_meta=True 時，support 只給模型建立 prototype/context；query 才計算 CE/reward。
                # 例：support 中的 SNHL/WNL 樣本不進 loss，query 中的樣本才會和真實 label 計算 CrossEntropy。
                support = None
                X_query, ys_query, raws_query = X, ys, raws
                if enable_meta:
                    support, query_idx = build_episode(
                        X,
                        ys,
                        label_cols,
                        max_support_per_class=support_per_class,
                    )
                    X_query, ys_query, raws_query = select_query_batch(X, ys, raws, query_idx)

                X_query = apply_training_missingness_augmentation(
                    X_query,
                    task_name,
                    union_features,
                    norm_meta[task_name],
                    missing_aug_p=missing_aug_p,
                    missing_aug_strategy_weights=missing_aug_strategy_weights,
                )

                logits_dict, reward_dict = model(X_query, task_name, support=support)

                ce_loss = 0.0
                reward_loss = 0.0
                for lbl, y in zip(label_cols, ys_query):
                    logits = logits_dict[lbl]

                    # 主分類損失
                    ce_loss = ce_loss + criteria_dict[lbl](logits, y)

                    # reward 頭：將目前預測映射成臨床一致性分數 target
                    probs = torch.softmax(logits.detach(), dim=-1)
                    pred_idx = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
                    pred_names = [tasks_meta[task_name]["class_names"][lbl][i] for i in pred_idx]
                    target_conf_pairs = [
                        expert_consistency_target_and_confidence(task_name, lbl, raw, pred_name)
                        for raw, pred_name in zip(raws_query, pred_names)
                    ]
                    tgt = [pair[0] for pair in target_conf_pairs]
                    conf = [pair[1] for pair in target_conf_pairs]
                    reward_targets = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)
                    reward_confidence = torch.tensor(conf, dtype=torch.float32, device=DEVICE)
                    reward_mse = nn.functional.mse_loss(
                        reward_dict[lbl],
                        reward_targets,
                        reduction="none",
                    )
                    reward_loss = reward_loss + (
                        reward_mse * reward_confidence
                    ).sum() / reward_confidence.sum().clamp_min(1e-6)

                # 不確定性加權 + reward loss
                log_var = model.log_vars[task_name]
                task_loss = torch.exp(-log_var) * ce_loss + log_var + reward_weight * reward_loss
                if torch.isnan(task_loss):
                    print(f"[WARN] NaN loss at task {task_name}, skip batch")
                    continue

                optimizer.zero_grad()
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                train_loss_total += float(task_loss.item())
                train_steps += 1
                task_step_counts[task_name] += 1

        # ===== 每個 epoch 結束後做驗證 =====
        val_summary = {}
        cm_summary = {}
        score_accum = 0.0
        score_count = 0
        for task_name in task_sequence:
            metrics, cms = evaluate_task(
                model,
                loaders_val[task_name],
                {"tasks": tasks_meta},
                task_name,
                criteria_by_task[task_name],
                use_meta_support=use_eval_meta_support,
                support_per_class=support_per_class,
            )
            val_summary[task_name] = metrics
            cm_summary[task_name] = cms
            for lbl in tasks_meta[task_name]["label_cols"]:
                score_accum += metrics.get(f"{lbl}_macro_f1", 0.0)
                score_count += 1

        # 用所有 label 的 macro-F1 平均作為最佳模型依據
        macro_score = score_accum / max(score_count, 1)
        val_loss = float(np.mean([val_summary[t]["loss"] for t in val_summary])) if val_summary else 0.0
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(train_steps, 1),
            "val_loss": val_loss,
            "mean_macro_f1": macro_score,
        }
        # NOTE 2026-05-20 P2-14:
        # 將每個 epoch 的 task 實際更新步數寫進 training_history.csv，方便事後確認 equal_steps 是否真的生效。
        for task_name, step_count in task_step_counts.items():
            row[f"{task_name}_train_steps"] = int(step_count)
        history_rows.append(row)
        print(f"[Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} mean_macro_f1={row['mean_macro_f1']:.4f} steps={task_step_counts}")

        macro_improved = macro_score > best_score
        loss_improved = val_loss < best_val_loss - min_delta

        if loss_improved:
            best_val_loss = val_loss
            bad_loss_epochs = 0
        else:
            bad_loss_epochs += 1

        # ===== 儲存最佳模型 =====
        # best checkpoint 仍以 macro-F1 為主；early stopping 才同時參考 val_loss。
        if macro_improved:
            best_score = macro_score
            bad_macro_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_meta = {
                "union_features": union_features,
                "union_features_count": len(union_features),
                "task_feature_counts": {
                    task: len(task_info["feature_cols"])
                    for task, task_info in TASK_INFO.items()
                },
                "feature_version": "v74_p4_task2_fourfreq_rule_no_mean_features",
                "tasks": tasks_meta,
                "norm_meta": norm_meta,
                "history": history_rows,
                "best_epoch": epoch,
                "best_mean_macro_f1": float(best_score),
                "best_val_loss_for_early_stop": float(best_val_loss),
                "train_sizes": train_sizes,
                "test_sizes": test_sizes,
                "cleaning_log": cleaning_log,
                "single_task_target": single_task_target,
                "version": "MetaIRL_v7.4",
                "exp_name": exp_name,
                "seed": seed,
                "enable_meta": enable_meta,
                "reward_weight": reward_weight,
                "reward_confidence": {
                    "task2_uncertain": DEFAULT_TASK2_UNCERTAIN_REWARD_CONFIDENCE,
                    "partial_rule": DEFAULT_PARTIAL_RULE_REWARD_CONFIDENCE,
                    "floor": DEFAULT_REWARD_CONFIDENCE_FLOOR,
                },
                "missingness_augmentation": {
                    "training_only": True,
                    "p": float(missing_aug_p),
                    "tasks": ["Task1", "Task2", "Task3"],
                    "task1_scenarios": ["mask_pta", "mask_high_freq", "mask_low_freq", "mask_all_ac"],
                    "task2_scenarios": ["no_bc", "partial_bc", "multi_ac_nr"],
                    "task3_scenarios": ["no_peak", "no_width", "np_like", "no_tymp"],
                    "strategy_weights": missing_aug_strategy_weights,
                    "profile": run_cfg.get("missing_aug_profile"),
                },
                "task_sequence": task_sequence,
                "model_config": model_config,
                # NOTE 2026-05-20 P2-12/P2-13/P2-14:
                # checkpoint metadata 會保存模型容量、task sampling 與 meta episode 設定。
                # 例：比較 run_01/run_07_small/run_09_tiny 時，可追溯當時是 128/4、64/2 或 32/1，且是否使用 equal_steps 與 validation support。
                "task_sampling": {
                    "mode": task_sampling_mode,
                    "configured_steps_per_task": steps_per_task,
                    "effective_steps_per_task": effective_steps_per_task,
                    "task_step_schedule_preview": task_step_schedule[:24],
                    "task_step_schedule_total_steps": len(task_step_schedule),
                },
                "meta_episode": {
                    "support_per_class": support_per_class,
                    "validate_with_meta_support": validate_with_meta_support,
                    "use_eval_meta_support": use_eval_meta_support,
                },
                "run_config": run_cfg,
            }
            ckpt = {"model_state_dict": best_state, "meta": best_meta}
            save_path = run_dir / "best_model.pth"
            torch.save(ckpt, save_path)
            print(f"[MetaIRL v7.4] Saved best checkpoint → {save_path}")
        else:
            bad_macro_epochs += 1

        if bad_macro_epochs >= patience and bad_loss_epochs >= patience:
            print(
                f"[MetaIRL v7.4] Early stopping at epoch {epoch} "
                f"(macro no improve={bad_macro_epochs}, loss no improve={bad_loss_epochs})"
            )
            break

    # 訓練歷史另存 CSV
    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "training_history.csv"
    hist_df.to_csv(hist_path, index=False)

    # 重新載入最佳權重做最終評估
    if best_state is not None:
        model.load_state_dict(best_state)

    def run_final_evaluation(
        eval_loaders: Dict[str, DataLoader],
        use_meta_support: bool,
        evaluation_name: str,
        save_predictions: bool = False,
    ):
        summary = {}
        matrices = {}
        prediction_frames = []
        for task_name in task_sequence:
            if task_name not in eval_loaders:
                continue
            if save_predictions:
                metrics, cms, pred_df = evaluate_task(
                    model,
                    eval_loaders[task_name],
                    {"tasks": tasks_meta},
                    task_name,
                    criteria_by_task[task_name],
                    use_meta_support=use_meta_support,
                    support_per_class=support_per_class,
                    return_predictions=True,
                )
                if pred_df is not None and len(pred_df) > 0:
                    pred_df.insert(0, "evaluation_mode", evaluation_name)
                    pred_df.insert(1, "use_meta_support", bool(use_meta_support))
                    pred_df.insert(2, "support_per_class", int(support_per_class if use_meta_support else 0))
                    prediction_frames.append(pred_df)
            else:
                metrics, cms = evaluate_task(
                    model,
                    eval_loaders[task_name],
                    {"tasks": tasks_meta},
                    task_name,
                    criteria_by_task[task_name],
                    use_meta_support=use_meta_support,
                    support_per_class=support_per_class,
                )
            summary[task_name] = metrics
            matrices[task_name] = cms
        pred_all = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        return summary, matrices, pred_all

    eval_summary, confusion, prediction_rows = run_final_evaluation(
        loaders_val,
        use_eval_meta_support,
        "configured_validation",
        save_predictions=True,
    )
    eval_summary_no_support, confusion_no_support, prediction_rows_no_support = run_final_evaluation(
        loaders_val,
        False,
        "deployment_no_support",
        save_predictions=True,
    )
    eval_summary_locked_test_no_support = {}
    confusion_locked_test_no_support = {}
    prediction_rows_locked_test_no_support = pd.DataFrame()
    if loaders_test:
        (
            eval_summary_locked_test_no_support,
            confusion_locked_test_no_support,
            prediction_rows_locked_test_no_support,
        ) = run_final_evaluation(
            loaders_test,
            False,
            "locked_test_no_support",
            save_predictions=True,
        )

    save_json({
        "history": history_rows,
        "evaluation_mode": {
            "name": "configured_validation",
            "use_meta_support": use_eval_meta_support,
            "support_per_class": support_per_class,
        },
        "eval_summary": eval_summary,
        "confusion_matrices": confusion,
        "cleaning_log": cleaning_log,
    }, run_dir / "eval_summary.json")
    save_json({
        "history": history_rows,
        "evaluation_mode": {
            "name": "deployment_no_support",
            "use_meta_support": False,
            "support_per_class": 0,
        },
        "eval_summary": eval_summary_no_support,
        "confusion_matrices": confusion_no_support,
        "cleaning_log": cleaning_log,
    }, run_dir / "eval_summary_no_support.json")
    if eval_summary_locked_test_no_support:
        save_json({
            "history": history_rows,
            "evaluation_mode": {
                "name": "locked_test_no_support",
                "use_meta_support": False,
                "support_per_class": 0,
                "source": "split_manifest_test_rows",
            },
            "eval_summary": eval_summary_locked_test_no_support,
            "confusion_matrices": confusion_locked_test_no_support,
            "cleaning_log": cleaning_log,
            "test_sizes": test_sizes,
        }, run_dir / "eval_summary_locked_test_no_support.json")
    # NOTE 2026-05-20 P2-15:
    # 將 eval_summary.json 中巢狀的 per-class metrics 另存成表格，方便直接檢查少數類別。
    # 例：用 Excel/Sheets 篩選 Task2 + CHL，就能看 precision/recall/F1/support。
    save_per_class_metrics_csv(eval_summary, run_dir / "per_class_metrics.csv")
    save_per_class_metrics_csv(eval_summary_no_support, run_dir / "per_class_metrics_no_support.csv")
    if eval_summary_locked_test_no_support:
        save_per_class_metrics_csv(
            eval_summary_locked_test_no_support,
            run_dir / "per_class_metrics_locked_test_no_support.csv",
        )
    prediction_rows.to_csv(run_dir / "prediction_rows.csv", index=False, encoding="utf-8-sig")
    prediction_rows_no_support.to_csv(run_dir / "prediction_rows_no_support.csv", index=False, encoding="utf-8-sig")
    if len(prediction_rows_locked_test_no_support):
        prediction_rows_locked_test_no_support.to_csv(
            run_dir / "prediction_rows_locked_test_no_support.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("[MetaIRL v7.4] Training finished.")
    print(f"[MetaIRL v7.4] Best macro-F1 = {best_score:.4f}")
    return {
        "exp_name": exp_name,
        "seed": seed,
        "best_mean_macro_f1": float(best_score),
        "run_dir": str(run_dir),
        "task_sequence": task_sequence,
        "single_task_target": single_task_target,
        "has_locked_test_evaluation": bool(eval_summary_locked_test_no_support),
    }


def aggregate_results_from_eval_file(
    root: Path,
    eval_filename: str,
    all_runs_filename: str,
    summary_filename: str,
    summary_json_filename: str,
):
    """Aggregate one evaluation mode into run-level and grouped summaries."""
    root = Path(root)
    rows = []
    for run_dir in sorted(root.glob("*_seed_*")):
        summary_path = run_dir / eval_filename
        if not summary_path.exists():
            continue
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        exp_name = run_dir.name.split("_seed_")[0]
        seed = int(run_dir.name.split("_seed_")[1])
        row = {"exp_name": exp_name, "seed": seed}
        for task_name, metrics in obj["eval_summary"].items():
            for k, v in metrics.items():
                row[f"{task_name}__{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return None

    # 所有 run 的原始 metrics
    df.to_csv(root / all_runs_filename, index=False)

    # 再依 exp_name 做平均與標準差
    grouped_rows = []
    numeric_cols = [c for c in df.columns if c not in ["exp_name", "seed"]]
    for exp_name, sub in df.groupby("exp_name"):
        g = {"exp_name": exp_name, "n_runs": int(len(sub))}
        for c in numeric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            g[f"{c}__mean"] = float(vals.mean()) if vals.notna().any() else None
            g[f"{c}__std"] = float(vals.std(ddof=0)) if vals.notna().any() else None
        grouped_rows.append(g)

    gdf = pd.DataFrame(grouped_rows)
    gdf.to_csv(root / summary_filename, index=False)
    save_json(grouped_rows, root / summary_json_filename)
    return gdf


def aggregate_results(out_root: str):
    """整理所有 seed / 實驗設定的結果，輸出 configured 與 no-support 總表。"""
    root = Path(out_root)
    configured = aggregate_results_from_eval_file(
        root,
        "eval_summary.json",
        "all_runs_metrics.csv",
        "summary.csv",
        "summary.json",
    )
    aggregate_results_from_eval_file(
        root,
        "eval_summary_no_support.json",
        "all_runs_metrics_no_support.csv",
        "summary_no_support.csv",
        "summary_no_support.json",
    )
    aggregate_results_from_eval_file(
        root,
        "eval_summary_locked_test_no_support.json",
        "all_runs_metrics_locked_test_no_support.csv",
        "summary_locked_test_no_support.csv",
        "summary_locked_test_no_support.json",
    )
    return configured


def parse_args():
    """解析命令列參數。"""
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--results_dir", type=str, default="results_v74")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--exp", type=str, default=None, choices=EXPERIMENT_CHOICES)
    p.add_argument("--experiments", type=str, default="full,no_meta,no_irl,single_task")
    p.add_argument(
        "--single_task_target",
        type=str,
        default="Task2",
        choices=["Task1", "Task2", "Task3", "all"],
        help="Target task for --exp single_task. Use all to expand to single_task_task1/task2/task3.",
    )
    p.add_argument(
        "--config_preset",
        type=str,
        default="recommended",
        choices=["recommended", "ablation", "all"],
        help="recommended = run_configs; ablation = ablation_run_configs; all = both. Full v1 JSON has 15+5 configs; v1_split JSON can select a subset.",
    )
    p.add_argument(
        "--run_config_file",
        "--run-config-file",
        type=str,
        default=str(DEFAULT_RUN_CONFIG_FILE),
        help="External v1 JSON or v1_split JSON selecting a subset from a base v1 JSON.",
    )
    p.add_argument(
        "--split_manifest",
        type=str,
        default=None,
        help="Optional CSV produced by split_protocol_v74.py. When set, train/val splits are loaded from the manifest.",
    )
    p.add_argument(
        "--locked_split_manifest",
        type=str,
        default=None,
        help="Alias for --split_manifest when using split_protocol_v74.py locked test manifests.",
    )
    p.add_argument(
        "--task_sampling_mode_override",
        type=str,
        default=None,
        choices=["full_loader", "equal_steps", "round_robin_k1", "round_robin_k2", "round_robin_k4", "round_robin_k16"],
        help="Override task_sampling_mode for all RUN_CONFIGS without changing hyperparameter sets.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 單一 seed 與多 seed 兩種模式
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # 單一實驗與多實驗兩種模式
    if args.exp is not None:
        experiments = [args.exp]
    else:
        experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    experiments = expand_experiment_names(experiments, args.single_task_target)

    config_sets, loaded_config_path = load_run_config_sets(args.run_config_file)
    if args.config_preset not in config_sets:
        raise ValueError(f"Unknown config_preset: {args.config_preset}")
    selected_config_source = config_sets[args.config_preset]
    print(
        f"[MetaIRL v7.4] Loaded run configs from {loaded_config_path} "
        f"preset={args.config_preset} n={len(selected_config_source)}"
    )

    run_configs = []
    for cfg in selected_config_source:
        cfg_effective = dict(cfg)
        if args.task_sampling_mode_override is not None:
            cfg_effective["task_sampling_mode"] = args.task_sampling_mode_override
        run_configs.append(cfg_effective)
    if args.split_manifest and args.locked_split_manifest:
        raise ValueError("Use only one of --split_manifest or --locked_split_manifest.")
    split_manifest_path = args.split_manifest or args.locked_split_manifest
    split_manifest = load_split_manifest(split_manifest_path)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.results_dir) / "five_runs"
    reset_runs_root(runs_root)
    save_run_configs_csv(run_configs, runs_root / "run_configs.csv")

    # 保留原本 5 組超參數；如有指定 override，僅覆寫 task sampling mode。
    for run_cfg in run_configs:
        cfg_root = runs_root / run_cfg["name"]
        cfg_root.mkdir(parents=True, exist_ok=True)
        save_run_configs_csv([run_cfg], cfg_root / "run_config.csv")
        all_runs = []

        # 逐個實驗、逐個 seed 執行
        for exp_name in experiments:
            for seed in seeds:
                run_info = run_experiment(
                    args.data_dir,
                    str(cfg_root),
                    seed,
                    exp_name,
                    run_cfg,
                    split_manifest=split_manifest,
                    single_task_target=args.single_task_target,
                )
                all_runs.append(run_info)

        # 每一組設定各自輸出自己的 CSV 與 summary
        save_json(all_runs, cfg_root / "run_manifest.json")
        aggregate_results(str(cfg_root))

    print(f"[MetaIRL v7.4] All runs finished. Results → {runs_root.resolve()}")


if __name__ == "__main__":
    main()
