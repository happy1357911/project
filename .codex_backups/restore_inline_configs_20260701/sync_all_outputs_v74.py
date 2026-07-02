import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_FILES = [
    "README_v74.txt",
    "run_all_v74.py",
    "run_locked_test_v74.py",
    "artificial_missingness_v74.py",
    "baseline_rules_v74.py",
    "calibration_analysis_v74.py",
    "checkpoint_utils_v74.py",
    "clinical_rules_v74.py",
    "dashboard_three_tasks_metaIRL_v74.py",
    "error_analysis_v74.py",
    "evaluate_v74.py",
    "feature_importance_v74.py",
    "hybrid_evaluation_v74.py",
    "ml_baselines_v74.py",
    "model_profile_v74.py",
    "mtl_meta_irl_transformer.py",
    "preprocessing_v74.py",
    "split_protocol_v74.py",
    "sync_all_outputs_v74.py",
    "train_v74.py",
    "configs/run_configs_v74.json",
    "configs/splits/run_configs_v74_part_A.json",
    "configs/splits/run_configs_v74_part_B.json",
    "paper_v74/manifest.json",
    "requirements.txt",
    "rule.md",
    "\u5c0d\u8a71\u7d00\u9304.md",
    "modification_history.md",
    "train_v74_\u5b8c\u6574\u8a13\u7df4\u6d41\u7a0b\u5716.md",
    "\u5c08\u6848\u67b6\u69cb\u8207\u6a21\u578b\u4f7f\u7528\u8aaa\u660e.md",
    "\u8207\u539f\u59cb\u7a0b\u5f0f\u78bc\u5dee\u7570\u6bd4\u5c0d.md",
    "\u8f38\u51fa\u6a94\u6848Label\u8207\u6b04\u4f4d\u610f\u7fa9\u8aaa\u660e.md",
]
DEFAULT_DIRS = [
    "paper_v74/tables",
    "paper_v74/five_runs",
    "paper_v74/error_analysis",
    "paper_v74/hybrid_evaluation",
    "results_v74/ml_baselines",
    "results_v74/split_protocol",
    "results_v74/rule_baselines_phase3",
    "results_v74/artificial_missingness",
    "results_v74/feature_importance",
    "results_v74/calibration_analysis",
    "results_v74/calibration_analysis_locked_test",
]
OPTIONAL_DIRS = [
    "paper_v74_locked_test/tables",
    "paper_v74_locked_test/hybrid_evaluation",
    "paper_v74_locked_test/error_analysis",
    "results_v74_locked_test/ml_baselines",
    "results_v74_locked_test/rule_baselines_phase3",
    "results_v74_locked_test/calibration_analysis",
    "results_v74_locked_test/model_profile",
]

LARGE_PREDICTION_FILES = {
    "prediction_rows_all.csv",
    "hybrid_predictions.csv",
    "hybrid_predictions_locked_test.csv",
    "artificial_missingness_predictions.csv",
    "feature_group_permutation_detail.csv",
}

CHECKPOINT_PATTERNS = ("*.pth", "*.pt", "*.ckpt")
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
COMPACT_FIVE_RUNS_DIR = "results_v74/five_runs"
COMPACT_FIVE_RUNS_FILES = {
    "all_runs_metrics.csv",
    "all_runs_metrics_locked_test_no_support.csv",
    "all_runs_metrics_no_support.csv",
    "run_config.csv",
    "run_manifest.json",
    "summary.csv",
    "summary.json",
    "summary_locked_test_no_support.csv",
    "summary_locked_test_no_support.json",
    "summary_no_support.csv",
    "summary_no_support.json",
}
COMPACT_FIVE_RUNS_ROOT_FILES = {
    "run_configs.csv",
}
RECORD_EXTENSION_FILES_EXCLUDED = [
    "cleanup_candidates_20260618.md",
]
RAW_DATA_FILES = [
    "task1_all_three_common14_v1.csv",
    "task2_3_pure_data(6_24).xlsx",
]

PACKAGE_TYPE_NOTES = {
    "analysis_only": (
        "GPT/professor review package: source code, documentation, formal tables, "
        "figures, compact outputs, and optionally active raw input files. "
        "Check include_raw_data/raw_data_scope, include_checkpoints, and "
        "include_large_prediction_rows before treating it as re-runnable."
    ),
    "execution_ready": (
        "Re-run package: includes raw input data, checkpoints, and large row-level "
        "prediction files when available. This package can be much larger."
    ),
    "custom": (
        "Custom package assembled from explicit include/exclude flags; inspect the "
        "manifest booleans before assuming whether it is re-runnable."
    ),
}


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_all_dir(root: Path, all_dir: Path) -> Path:
    return all_dir if all_dir.is_absolute() else root / all_dir


def clean_all_dir(root: Path, all_dir: Path) -> None:
    root_resolved = root.resolve()
    target = all_dir.resolve() if all_dir.exists() else all_dir.absolute()
    if target == root_resolved:
        raise ValueError("Refusing to clean project root as all_dir.")
    if not _is_relative_to(target, root_resolved):
        raise ValueError(f"Refusing to clean a target outside project root: {target}")
    if all_dir.exists():
        shutil.rmtree(all_dir)


def copy_file(src: Path, dst: Path, *, include_checkpoints: bool) -> bool:
    if not src.exists():
        return False
    if src.suffix.lower() in CHECKPOINT_SUFFIXES and not include_checkpoints:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_dir(src: Path, dst: Path, *, skip_large_predictions: bool, include_checkpoints: bool) -> bool:
    if not src.exists():
        return False
    patterns = []
    if skip_large_predictions:
        patterns.extend(sorted(LARGE_PREDICTION_FILES))
    if not include_checkpoints:
        patterns.extend(CHECKPOINT_PATTERNS)
    ignore = shutil.ignore_patterns(*patterns) if patterns else None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    return True


def copy_compact_five_runs(src: Path, dst: Path) -> list[str]:
    if not src.exists():
        return []
    copied = []
    dst.mkdir(parents=True, exist_ok=True)

    for file_name in sorted(COMPACT_FIVE_RUNS_ROOT_FILES):
        src_file = src / file_name
        if src_file.exists() and src_file.is_file():
            shutil.copy2(src_file, dst / file_name)
            copied.append(file_name)

    for config_dir in sorted(path for path in src.iterdir() if path.is_dir()):
        target_dir = dst / config_dir.name
        for file_name in sorted(COMPACT_FIVE_RUNS_FILES):
            src_file = config_dir / file_name
            if not src_file.exists() or not src_file.is_file():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, target_dir / file_name)
            copied.append(str(Path(config_dir.name) / file_name))
    return copied


def inventory_all_dir(all_dir: Path) -> dict:
    files = [path for path in all_dir.rglob("*") if path.is_file()]
    checkpoint_files = [str(path.relative_to(all_dir)) for path in files if path.suffix.lower() in CHECKPOINT_SUFFIXES]
    large_prediction_files = [str(path.relative_to(all_dir)) for path in files if path.name in LARGE_PREDICTION_FILES]
    return {
        "file_count": len(files),
        "directory_count": sum(1 for path in all_dir.rglob("*") if path.is_dir()),
        "checkpoint_files_in_all": checkpoint_files,
        "large_prediction_files_in_all": large_prediction_files,
    }


def sync_all(
    root: Path,
    all_dir: Path,
    *,
    package_type: str = "analysis_only",
    skip_large_predictions: bool = False,
    include_checkpoints: bool = False,
    include_raw_data: bool = False,
    clean: bool = False,
    excluded_dirs: list[str] | None = None,
):
    root = root.resolve()
    all_dir = resolve_all_dir(root, all_dir)
    if clean:
        clean_all_dir(root, all_dir)
    all_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    copied_dirs = []
    missing = []
    skipped_checkpoint_files = []
    excluded_dir_set = {str(item).replace("\\", "/").strip("/") for item in (excluded_dirs or []) if str(item).strip()}

    for rel in DEFAULT_FILES:
        src = root / rel
        if src.suffix.lower() in CHECKPOINT_SUFFIXES and not include_checkpoints:
            skipped_checkpoint_files.append(rel)
            continue
        if copy_file(src, all_dir / rel, include_checkpoints=include_checkpoints):
            copied_files.append(rel)
        else:
            missing.append(rel)

    raw_data_copied = []
    raw_data_missing = []
    if include_raw_data:
        for rel in RAW_DATA_FILES:
            if copy_file(root / rel, all_dir / rel, include_checkpoints=include_checkpoints):
                copied_files.append(rel)
                raw_data_copied.append(rel)
            else:
                raw_data_missing.append(rel)
                missing.append(rel)

    for rel in DEFAULT_DIRS:
        rel_key = rel.replace("\\", "/").strip("/")
        if rel_key in excluded_dir_set:
            continue
        if copy_dir(root / rel, all_dir / rel, skip_large_predictions=skip_large_predictions, include_checkpoints=include_checkpoints):
            copied_dirs.append(rel)
        else:
            missing.append(rel)

    compact_five_runs_files = []
    five_runs_sync_mode = "excluded"
    five_runs_key = COMPACT_FIVE_RUNS_DIR.replace("\\", "/").strip("/")
    if five_runs_key not in excluded_dir_set:
        five_runs_src = root / COMPACT_FIVE_RUNS_DIR
        five_runs_dst = all_dir / COMPACT_FIVE_RUNS_DIR
        if include_checkpoints:
            five_runs_sync_mode = "full"
            if copy_dir(
                five_runs_src,
                five_runs_dst,
                skip_large_predictions=skip_large_predictions,
                include_checkpoints=include_checkpoints,
            ):
                copied_dirs.append(COMPACT_FIVE_RUNS_DIR)
            else:
                missing.append(COMPACT_FIVE_RUNS_DIR)
        else:
            five_runs_sync_mode = "compact_summary_only"
            compact_five_runs_files = copy_compact_five_runs(five_runs_src, five_runs_dst)
            if compact_five_runs_files:
                copied_dirs.append(COMPACT_FIVE_RUNS_DIR)
            else:
                missing.append(COMPACT_FIVE_RUNS_DIR)

    optional_copied_dirs = []
    for rel in OPTIONAL_DIRS:
        rel_key = rel.replace("\\", "/").strip("/")
        if rel_key in excluded_dir_set:
            continue
        if copy_dir(root / rel, all_dir / rel, skip_large_predictions=skip_large_predictions, include_checkpoints=include_checkpoints):
            copied_dirs.append(rel)
            optional_copied_dirs.append(rel)

    inventory = inventory_all_dir(all_dir)
    include_large_prediction_rows = not skip_large_predictions
    is_execution_ready = bool(include_raw_data and include_checkpoints and include_large_prediction_rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "package_type": package_type,
        "package_type_note": PACKAGE_TYPE_NOTES.get(package_type, PACKAGE_TYPE_NOTES["custom"]),
        "is_execution_ready": is_execution_ready,
        "include_raw_data": bool(include_raw_data),
        "include_large_prediction_rows": include_large_prediction_rows,
        "root": str(root),
        "all_dir": str(all_dir.resolve()),
        "clean": bool(clean),
        "skip_large_predictions": bool(skip_large_predictions),
        "include_checkpoints": bool(include_checkpoints),
        "checkpoint_exclusion_patterns": list(CHECKPOINT_PATTERNS) if not include_checkpoints else [],
        "large_prediction_exclusion_names": sorted(LARGE_PREDICTION_FILES) if skip_large_predictions else [],
        "five_runs_sync_mode": five_runs_sync_mode,
        "compact_five_runs_dir": COMPACT_FIVE_RUNS_DIR,
        "compact_five_runs_allowed_files": sorted(COMPACT_FIVE_RUNS_FILES),
        "compact_five_runs_root_files": sorted(COMPACT_FIVE_RUNS_ROOT_FILES),
        "compact_five_runs_files": compact_five_runs_files,
        "record_extension_files_excluded": list(RECORD_EXTENSION_FILES_EXCLUDED),
        "copied_files": copied_files,
        "copied_dirs": copied_dirs,
        "optional_copied_dirs": optional_copied_dirs,
        "raw_data_scope": "active_input_sources" if include_raw_data else "excluded",
        "raw_data_files_expected": list(RAW_DATA_FILES) if include_raw_data else [],
        "raw_data_files_in_all": raw_data_copied,
        "raw_data_missing": raw_data_missing,
        "missing_sources": missing,
        "skipped_checkpoint_files": skipped_checkpoint_files,
        "excluded_dirs": sorted(excluded_dir_set),
        **inventory,
    }
    manifest_path = all_dir / "sync_manifest_v74.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Sync selected v7.4 source files and formal outputs into all/.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--all-dir", default="all", help="Target all/ directory.")
    parser.add_argument("--clean", action="store_true", help="Safely clear all/ before syncing.")
    parser.add_argument("--package-type", default="analysis_only", choices=["analysis_only", "execution_ready", "custom"], help="Declare whether all/ is an analysis-only, execution-ready, or custom package.")
    parser.add_argument("--include-checkpoints", action="store_true", help="Include .pth/.pt/.ckpt files in all/. Off by default for GPT upload packages.")
    parser.add_argument("--include-raw-data", action="store_true", help="Include the active raw input data files in all/. Off by default for GPT upload packages.")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Relative output directory to omit from sync requirements; can be repeated.")
    parser.add_argument(
        "--skip-large-predictions",
        action="store_true",
        help="Skip large prediction CSV files when syncing paper outputs.",
    )
    args = parser.parse_args()
    manifest = sync_all(
        Path(args.root),
        Path(args.all_dir),
        package_type=args.package_type,
        skip_large_predictions=args.skip_large_predictions,
        include_checkpoints=args.include_checkpoints,
        include_raw_data=args.include_raw_data,
        clean=args.clean,
        excluded_dirs=args.exclude_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
