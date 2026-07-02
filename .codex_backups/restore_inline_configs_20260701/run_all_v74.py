import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from checkpoint_utils_v74 import resolve_checkpoint_paths, resolve_single_checkpoint


ALL_PY_FILES = [
    "artificial_missingness_v74.py",
    "checkpoint_utils_v74.py",
    "baseline_rules_v74.py",
    "calibration_analysis_v74.py",
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
    "run_all_v74.py",
    "run_locked_test_v74.py",
    "split_protocol_v74.py",
    "sync_all_outputs_v74.py",
    "train_v74.py",
]

EXECUTABLE_PIPELINE_SCRIPTS = [
    "train_v74.py",
    "baseline_rules_v74.py",
    "ml_baselines_v74.py",
    "error_analysis_v74.py",
    "evaluate_v74.py",
    "feature_importance_v74.py",
    "split_protocol_v74.py",
    "hybrid_evaluation_v74.py",
    "artificial_missingness_v74.py",
    "calibration_analysis_v74.py",
    "model_profile_v74.py",
    "run_locked_test_v74.py",
    "sync_all_outputs_v74.py",
]

SUPPORT_OR_INTERACTIVE_SCRIPTS = [
    "clinical_rules_v74.py",
    "preprocessing_v74.py",
    "mtl_meta_irl_transformer.py",
    "dashboard_three_tasks_metaIRL_v74.py",
    "run_all_v74.py",
]


def write_section(message: str) -> None:
    print()
    print("=" * 60)
    print(message)
    print("=" * 60)


def format_command_line(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def new_python_command(args: argparse.Namespace, python_args: list[str]) -> list[str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python", *python_args]
    return [args.python, *python_args]


def invoke_step(name: str, command: list[str], dry_run: bool) -> None:
    write_section(name)
    print(format_command_line(command))
    if dry_run:
        print("[DRY-RUN] skipped")
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit code {result.returncode})")


def invoke_local_step(name: str, action, dry_run: bool) -> None:
    write_section(name)
    if dry_run:
        print("[DRY-RUN] skipped")
        return
    action()


def primary_analysis_checkpoint_glob(args: argparse.Namespace) -> str:
    if args.analysis_checkpoint_glob:
        return args.analysis_checkpoint_glob
    config_pattern = args.primary_analysis_config if args.primary_analysis_config != "auto" else "*"
    exp_pattern = f"{args.primary_analysis_exp}_seed_*" if args.primary_analysis_exp != "auto" else "*_seed_*"
    return str(
        Path(args.results_dir)
        / "five_runs"
        / config_pattern
        / exp_pattern
        / "best_model.pth"
    )


def resolve_run_checkpoints(args: argparse.Namespace) -> list[str]:
    results_dir = Path(args.results_dir)
    if args.checkpoint:
        paths = resolve_checkpoint_paths(
            checkpoint=args.checkpoint,
            checkpoint_glob=None,
            default_checkpoint=None,
            search_defaults=False,
        )
        return [str(path.resolve()) for path in paths]

    checkpoint_glob = primary_analysis_checkpoint_glob(args)
    paths = [Path(path) for path in sorted(glob.glob(checkpoint_glob, recursive=True))]
    if not paths and args.primary_analysis_exp not in {"full", "auto"} and not args.analysis_checkpoint_glob:
        fallback_glob = str(results_dir / "five_runs" / "*" / "full_seed_*" / "best_model.pth")
        paths = [Path(path) for path in sorted(glob.glob(fallback_glob, recursive=True))]
        if paths:
            print(f"[WARN] No checkpoints matched {checkpoint_glob}; using fallback {fallback_glob}")
    if not paths:
        raise FileNotFoundError(f"No checkpoint resolved. Checked: {checkpoint_glob}")
    return [str(path.resolve()) for path in paths]


def join_checkpoints(paths: list[str]) -> str:
    return ",".join(paths)


def assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")


def verify_key_outputs(args: argparse.Namespace) -> None:
    paper_dir = Path(args.paper_dir)
    results_dir = Path(args.results_dir)
    all_dir = Path(args.all_dir)
    paper_five_runs_dir = paper_dir / "five_runs"
    all_paper_five_runs_dir = all_dir / "paper_v74" / "five_runs"
    all_results_five_runs_dir = all_dir / "results_v74" / "five_runs"

    assert_file_exists(paper_five_runs_dir)
    paper_run_dirs = [
        p for p in paper_five_runs_dir.iterdir()
        if p.is_dir() and (p.name.startswith("run_") or p.name.startswith("ablation_"))
    ]
    if not paper_run_dirs:
        raise FileNotFoundError(f"No paper per-config run directories found under {paper_five_runs_dir}.")
    flattened_paper_dirs = [
        p.name for p in paper_dir.iterdir()
        if p.is_dir() and (p.name.startswith("run_") or p.name.startswith("ablation_"))
    ]
    if flattened_paper_dirs:
        raise RuntimeError(
            "paper_v74 contains flattened run directories; expected paper_v74/five_runs only: "
            + ", ".join(sorted(flattened_paper_dirs))
        )

    assert_file_exists(paper_dir / "tables" / "summary_no_support_all_configs.csv")
    assert_file_exists(paper_dir / "tables" / "summary_locked_test_no_support_all_configs.csv")
    assert_file_exists(paper_dir / "tables" / "statistical_summary_all_configs.csv")
    assert_file_exists(paper_dir / "tables" / "main_hybrid_summary.csv")
    assert_file_exists(paper_dir / "tables" / "main_hybrid_summary_locked_test.csv")
    assert_file_exists(results_dir / "rule_baselines_phase3" / "rule_baseline_summary.csv")
    assert_file_exists(results_dir / "ml_baselines" / "ml_baseline_summary.csv")
    assert_file_exists(results_dir / "ml_baselines" / "ml_baseline_summary_5seed.csv")
    assert_file_exists(results_dir / "ml_baselines" / "ml_baseline_locked_test_summary_5seed.csv")
    assert_file_exists(paper_dir / "error_analysis" / "subgroup_metrics.csv")
    assert_file_exists(paper_dir / "error_analysis" / "rule_true_model_conflicts.csv")
    assert_file_exists(paper_dir / "error_analysis" / "three_way_conflict_summary.csv")
    assert_file_exists(paper_dir / "error_analysis" / "clinical_error_taxonomy_summary.csv")
    assert_file_exists(results_dir / "split_protocol" / "split_manifest.csv")
    assert_file_exists(results_dir / "split_protocol" / "split_summary.csv")
    if args.skip_calibration:
        print("[INFO] Calibration checks skipped because --skip-calibration is set.")
    else:
        assert_file_exists(results_dir / "calibration_analysis" / "calibration_summary.csv")
        assert_file_exists(results_dir / "calibration_analysis" / "calibration_ece_bins.csv")
        assert_file_exists(results_dir / "calibration_analysis" / "confidence_threshold_curve.csv")
        assert_file_exists(results_dir / "calibration_analysis" / "calibration_policy_summary.csv")
        assert_file_exists(results_dir / "calibration_analysis_locked_test" / "calibration_summary.csv")
        assert_file_exists(results_dir / "calibration_analysis_locked_test" / "calibration_ece_bins.csv")
        assert_file_exists(results_dir / "calibration_analysis_locked_test" / "confidence_threshold_curve.csv")
        assert_file_exists(results_dir / "calibration_analysis_locked_test" / "calibration_policy_summary.csv")
    if args.skip_artificial_missingness:
        print("[INFO] Artificial missingness checks skipped because --skip-artificial-missingness is set.")
    else:
        assert_file_exists(results_dir / "artificial_missingness" / "artificial_missingness_summary.csv")
        assert_file_exists(results_dir / "artificial_missingness" / "artificial_missingness_degradation_summary.csv")
        assert_file_exists(results_dir / "artificial_missingness" / "evidence_compensation_summary.csv")
    if args.skip_feature_importance:
        print("[INFO] Feature-importance checks skipped because --skip-feature-importance is set.")
    else:
        assert_file_exists(results_dir / "feature_importance" / "feature_importance_baseline.csv")
        assert_file_exists(results_dir / "feature_importance" / "feature_group_ablation_summary.csv")
        assert_file_exists(results_dir / "feature_importance" / "feature_group_permutation_importance.csv")
        assert_file_exists(results_dir / "feature_importance" / "feature_importance_manifest.json")
    if args.skip_model_profile:
        print("[INFO] Optional model/profile checks skipped because --skip-model-profile is set.")
    else:
        assert_file_exists(results_dir / "model_profile" / "model_profile.csv")
        if args.require_edge_profile:
            assert_file_exists(results_dir / "model_profile" / "model_profile_summary.csv")
            assert_file_exists(results_dir / "model_profile" / "deployment_profile.csv")
        else:
            print("[INFO] Optional deployment/profile summary checks deferred; use --require-edge-profile to enforce them.")
    assert_file_exists(paper_dir / "hybrid_evaluation" / "hybrid_summary.csv")
    assert_file_exists(paper_dir / "hybrid_evaluation" / "hybrid_locked_test_summary.csv")
    assert_file_exists(paper_dir / "hybrid_evaluation" / "hybrid_strategy_mcnemar.csv")
    assert_file_exists(paper_dir / "hybrid_evaluation" / "hybrid_strategy_mcnemar_locked_test.csv")
    if args.hybrid_threshold_sweep:
        assert_file_exists(paper_dir / "hybrid_evaluation" / "hybrid_threshold_sweep.csv")
        assert_file_exists(paper_dir / "hybrid_evaluation" / "hybrid_threshold_sweep_locked_test.csv")
    assert_file_exists(all_dir / "sync_manifest_v74.json")

    if args.run_locked_test:
        locked_paper = Path(args.locked_paper_dir)
        assert_file_exists(locked_paper / "tables" / "summary_locked_test_no_support_all_configs.csv")
        assert_file_exists(locked_paper / "tables" / "main_hybrid_summary_locked_test.csv")

    no_support_cm = list(paper_dir.glob("**/cm_*no_support.png"))
    if not no_support_cm:
        raise FileNotFoundError(f"No no-support confusion matrix images were found under {paper_dir}.")
    print(f"No-support confusion matrices found: {len(no_support_cm)}")

    manifest = json.loads((all_dir / "sync_manifest_v74.json").read_text(encoding="utf-8"))
    missing_sources = manifest.get("missing_sources") or []
    if missing_sources:
        raise RuntimeError(f"all/ sync has missing sources: {', '.join(missing_sources)}")
    if manifest.get("package_type") != args.package_type:
        raise RuntimeError(
            f"all/ package_type mismatch: expected {args.package_type}, got {manifest.get('package_type')}"
        )
    expected_raw_data = bool(args.include_raw_data or args.package_type == "execution_ready")
    expected_checkpoints = bool(args.include_checkpoints or args.package_type == "execution_ready")
    expected_large_rows = bool(args.include_large_prediction_rows or args.package_type == "execution_ready")
    if bool(manifest.get("include_raw_data")) != expected_raw_data:
        raise RuntimeError("all/ include_raw_data does not match run_all arguments")
    if bool(manifest.get("include_checkpoints")) != expected_checkpoints:
        raise RuntimeError("all/ include_checkpoints does not match run_all arguments")
    if bool(manifest.get("include_large_prediction_rows")) != expected_large_rows:
        raise RuntimeError("all/ include_large_prediction_rows does not match run_all arguments")
    expected_five_runs_mode = "full" if expected_checkpoints else "compact_summary_only"
    if manifest.get("five_runs_sync_mode") != expected_five_runs_mode:
        raise RuntimeError(
            f"all/ five_runs_sync_mode mismatch: expected {expected_five_runs_mode}, "
            f"got {manifest.get('five_runs_sync_mode')}"
        )
    assert_file_exists(all_paper_five_runs_dir)
    assert_file_exists(all_results_five_runs_dir / "run_configs.csv")
    if not expected_checkpoints and manifest.get("checkpoint_files_in_all"):
        raise RuntimeError("all/ contains checkpoint files, but checkpoints were not requested")
    print("all/ sync missing_sources=[]")
    print(
        "all/ package flags: "
        f"package_type={manifest.get('package_type')}, "
        f"raw_data={manifest.get('include_raw_data')}, "
        f"checkpoints={manifest.get('include_checkpoints')}, "
        f"large_prediction_rows={manifest.get('include_large_prediction_rows')}"
    )


def print_file_coverage() -> None:
    write_section("Python file coverage")
    print("All root .py files checked by py_compile:")
    for file_name in ALL_PY_FILES:
        print(f"  - {file_name}")
    print()
    print("Executable pipeline scripts:")
    for file_name in EXECUTABLE_PIPELINE_SCRIPTS:
        print(f"  - {file_name}")
    print()
    print("Support / interactive scripts checked by compile:")
    for file_name in SUPPORT_OR_INTERACTIVE_SCRIPTS:
        print(f"  - {file_name}")

    missing = [file_name for file_name in ALL_PY_FILES if not Path(file_name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Python files: {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v7.4 missing-aware clinical-rule-guided hearing decision-support pipeline.")
    parser.add_argument("--python", default="python", help="Python executable used when --conda-env is empty.")
    parser.add_argument("--conda-env", default="", help="Optional conda environment name.")
    parser.add_argument("--data-dir", default=".", help="Input data directory.")
    parser.add_argument("--results-dir", default="results_v74", help="Training/results output directory.")
    parser.add_argument("--paper-dir", default="paper_v74", help="Paper tables/figures output directory.")
    parser.add_argument("--all-dir", default="all", help="Synced formal package directory.")
    parser.add_argument("--locked-results-dir", default="results_v74_locked_test")
    parser.add_argument("--locked-paper-dir", default="paper_v74_locked_test")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds.")
    parser.add_argument("--config-preset", default="all", help="train_v74.py config preset.")
    parser.add_argument("--run-config-file", default="configs/run_configs_v74.json", help="External JSON run config file passed to train_v74.py and locked-test training.")
    parser.add_argument("--experiments", default="full,no_meta,no_irl,single_task")
    parser.add_argument("--single-task-target", default="all")
    parser.add_argument("--heatmap-mode", default="seed0")
    parser.add_argument("--locked-test-ratio", type=float, default=0.2)
    parser.add_argument("--rule-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--model-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--checkpoint", default="", help="Checkpoint for artificial missingness/profile.")
    parser.add_argument("--primary-analysis-config", default="auto", help="Primary config used for missingness and feature-importance analysis.")
    parser.add_argument("--primary-analysis-exp", default="auto", help="Primary experiment family used for missingness and feature-importance analysis. Use auto to scan every available *_seed_* checkpoint under five_runs.")
    parser.add_argument("--analysis-checkpoint-glob", default="", help="Override checkpoint glob for missingness and feature-importance analysis.")
    parser.add_argument("--hybrid-threshold-sweep", action="store_true", default=True, help="Export hybrid threshold sweep tables.")
    parser.add_argument("--no-hybrid-threshold-sweep", dest="hybrid_threshold_sweep", action="store_false", help="Disable hybrid threshold sweep tables.")
    parser.add_argument("--hybrid-rule-thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--hybrid-model-thresholds", default="0.4,0.5,0.6,0.7,0.8,0.9")

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-rule-baseline", action="store_true")
    parser.add_argument("--skip-ml-baseline", action="store_true")
    parser.add_argument("--skip-error-analysis", action="store_true")
    parser.add_argument("--skip-split-protocol", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-artificial-missingness", action="store_true")
    parser.add_argument("--skip-feature-importance", action="store_true")
    parser.add_argument("--feature-importance-repeats", type=int, default=5)
    parser.add_argument("--skip-model-profile", action="store_true")
    parser.add_argument("--require-edge-profile", action="store_true", help="Require model_profile_summary.csv and deployment_profile.csv during final verification.")
    parser.add_argument("--skip-sync-all", action="store_true")
    parser.add_argument("--package-type", default="analysis_only", choices=["analysis_only", "execution_ready", "custom"], help="Package declaration written to all/sync_manifest_v74.json.")
    parser.add_argument("--include-raw-data", action="store_true", help="Include active raw data files in all/. Automatically enabled by --package-type execution_ready.")
    parser.add_argument("--include-checkpoints", action="store_true", help="Include .pth/.pt/.ckpt checkpoints in all/. Automatically enabled by --package-type execution_ready.")
    parser.add_argument("--include-large-prediction-rows", action="store_true", help="Include row-level prediction CSVs in all/. Automatically enabled by --package-type execution_ready.")
    parser.add_argument("--quick-ml-baseline", action="store_true")
    parser.add_argument("--profile-ml-baselines", action="store_true")
    parser.add_argument("--skip-profile-ml-baselines", action="store_true")
    parser.add_argument("--quick-profile-ml-baselines", action="store_true")
    parser.add_argument("--run-locked-test", action="store_true")
    parser.add_argument("--locked-allow-overwrite", action="store_true")
    parser.add_argument("--run-dashboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_file_coverage()

    invoke_step(
        "0. Compile every root Python file",
        new_python_command(args, ["-m", "py_compile", *ALL_PY_FILES]),
        args.dry_run,
    )

    if args.compile_only:
        write_section("compile-only completed")
        print("All root Python files were compiled successfully.")
        return 0

    split_manifest_path = Path(args.results_dir) / "split_protocol" / "split_manifest.csv"

    if not args.skip_split_protocol:
        invoke_step(
            "1. Grouped split protocol / locked-test manifest",
            new_python_command(
                args,
                [
                    "split_protocol_v74.py",
                    "--data_dir",
                    args.data_dir,
                    "--output_dir",
                    str(Path(args.results_dir) / "split_protocol"),
                    "--seeds",
                    args.seeds,
                    "--locked_test_ratio",
                    str(args.locked_test_ratio),
                ],
            ),
            args.dry_run,
        )
    else:
        write_section("1. Grouped split protocol / locked-test manifest")
        print("[SKIP] --skip-split-protocol was set; reusing existing split manifest")

    if not args.skip_train:
        invoke_step(
            "2. Train all selected v7.4 experiments with locked-test manifest",
            new_python_command(
                args,
                [
                    "train_v74.py",
                    "--data_dir",
                    args.data_dir,
                    "--results_dir",
                    args.results_dir,
                    "--seeds",
                    args.seeds,
                    "--experiments",
                    args.experiments,
                    "--single_task_target",
                    args.single_task_target,
                    "--config_preset",
                    args.config_preset,
                    "--run_config_file",
                    args.run_config_file,
                    "--locked_split_manifest",
                    str(split_manifest_path),
                ],
            ),
            args.dry_run,
        )
    else:
        write_section("2. Train all selected v7.4 experiments with locked-test manifest")
        print("[SKIP] --skip-train was set")

    if not args.skip_rule_baseline:
        invoke_step(
            "3. Missing-aware clinical rule baseline",
            new_python_command(
                args,
                [
                    "baseline_rules_v74.py",
                    "--data-dir",
                    args.data_dir,
                    "--output-dir",
                    str(Path(args.results_dir) / "rule_baselines_phase3"),
                ],
            ),
            args.dry_run,
        )

    if not args.skip_ml_baseline:
        ml_args = [
            "ml_baselines_v74.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(Path(args.results_dir) / "ml_baselines"),
            "--seeds",
            args.seeds,
        ]
        if args.quick_ml_baseline:
            ml_args.append("--quick")
        invoke_step("4a. Classical ML baseline / grouped validation", new_python_command(args, ml_args), args.dry_run)
        locked_ml_args = [
            *ml_args,
            "--eval-mode",
            "locked_test",
            "--split-manifest",
            str(split_manifest_path),
        ]
        invoke_step("4b. Classical ML baseline / locked-test", new_python_command(args, locked_ml_args), args.dry_run)

    if not args.skip_error_analysis:
        invoke_step(
            "5. Error / subgroup analysis",
            new_python_command(
                args,
                [
                    "error_analysis_v74.py",
                    "--results_dir",
                    args.results_dir,
                    "--output_dir",
                    str(Path(args.paper_dir) / "error_analysis"),
                    "--mode",
                    "all",
                    "--rule_predictions",
                    str(Path(args.results_dir) / "rule_baselines_phase3" / "rule_baseline_predictions.csv"),
                ],
            ),
            args.dry_run,
        )

    if not args.skip_evaluate:
        invoke_step(
            "6. Paper table and figure export",
            new_python_command(
                args,
                [
                    "evaluate_v74.py",
                    "--results_dir",
                    args.results_dir,
                    "--paper_dir",
                    args.paper_dir,
                    "--heatmap_mode",
                    args.heatmap_mode,
                ],
            ),
            args.dry_run,
        )

    if not args.skip_hybrid:
        invoke_step(
            "7a. Hybrid rule-first evaluation / no-support",
            new_python_command(
                args,
                [
                    "hybrid_evaluation_v74.py",
                    "--results_dir",
                    args.results_dir,
                    "--rule_predictions",
                    str(Path(args.results_dir) / "rule_baselines_phase3" / "rule_baseline_predictions.csv"),
                    "--output_dir",
                    str(Path(args.paper_dir) / "hybrid_evaluation"),
                    "--paper_tables_dir",
                    str(Path(args.paper_dir) / "tables"),
                    "--mode",
                    "no_support",
                    "--rule_confidence_threshold",
                    str(args.rule_confidence_threshold),
                    "--model_confidence_threshold",
                    str(args.model_confidence_threshold),
                    *( ["--threshold_sweep", "--rule_thresholds", args.hybrid_rule_thresholds, "--model_thresholds", args.hybrid_model_thresholds] if args.hybrid_threshold_sweep else [] ),
                ],
            ),
            args.dry_run,
        )
        invoke_step(
            "7b. Hybrid rule-first evaluation / locked-test",
            new_python_command(
                args,
                [
                    "hybrid_evaluation_v74.py",
                    "--results_dir",
                    args.results_dir,
                    "--rule_predictions",
                    str(Path(args.results_dir) / "rule_baselines_phase3" / "rule_baseline_predictions.csv"),
                    "--output_dir",
                    str(Path(args.paper_dir) / "hybrid_evaluation"),
                    "--paper_tables_dir",
                    str(Path(args.paper_dir) / "tables"),
                    "--mode",
                    "locked_test",
                    "--rule_confidence_threshold",
                    str(args.rule_confidence_threshold),
                    "--model_confidence_threshold",
                    str(args.model_confidence_threshold),
                    *( ["--threshold_sweep", "--rule_thresholds", args.hybrid_rule_thresholds, "--model_thresholds", args.hybrid_model_thresholds] if args.hybrid_threshold_sweep else [] ),
                ],
            ),
            args.dry_run,
        )

    if not args.skip_calibration:
        invoke_step(
            "8a. Calibration / confidence analysis",
            new_python_command(
                args,
                [
                    "calibration_analysis_v74.py",
                    "--prediction_rows",
                    str(Path(args.paper_dir) / "error_analysis" / "prediction_rows_all.csv"),
                    "--output_dir",
                    str(Path(args.results_dir) / "calibration_analysis"),
                    "--mode",
                    "both",
                    "--bins",
                    "10",
                    "--thresholds",
                    "0.0,0.5,0.6,0.7,0.8,0.9,0.95",
                ],
            ),
            args.dry_run,
        )
        invoke_step(
            "8b. Locked-test calibration / confidence analysis",
            new_python_command(
                args,
                [
                    "calibration_analysis_v74.py",
                    "--results_dir",
                    args.results_dir,
                    "--output_dir",
                    str(Path(args.results_dir) / "calibration_analysis_locked_test"),
                    "--mode",
                    "locked_test",
                    "--bins",
                    "10",
                    "--thresholds",
                    "0.0,0.5,0.6,0.7,0.8,0.9,0.95",
                ],
            ),
            args.dry_run,
        )

    resolved_checkpoints: list[str] = []
    analysis_checkpoint_args: list[str] = []
    needs_checkpoint = (not args.skip_artificial_missingness) or (not args.skip_feature_importance)
    if needs_checkpoint:
        if args.checkpoint:
            analysis_checkpoint_args = ["--checkpoint", args.checkpoint]
            checkpoint_input = args.checkpoint
        else:
            checkpoint_input = primary_analysis_checkpoint_glob(args)
            analysis_checkpoint_args = ["--checkpoint-glob", checkpoint_input]
        if args.dry_run:
            resolved_checkpoints = [checkpoint_input]
        else:
            resolved_checkpoints = resolve_run_checkpoints(args)
        print("Using analysis checkpoint input:")
        print(f"  {analysis_checkpoint_args[0]} {analysis_checkpoint_args[1]}")
        print(f"Resolved analysis checkpoints: {len(resolved_checkpoints)}")
        preview = resolved_checkpoints[:5]
        for checkpoint in preview:
            print(f"  - {checkpoint}")
        if len(resolved_checkpoints) > len(preview):
            print(f"  ... {len(resolved_checkpoints) - len(preview)} more checkpoints")


    if not args.skip_artificial_missingness:
        invoke_step(
            "9. Artificial missingness stress test",
            new_python_command(
                args,
                [
                    "artificial_missingness_v74.py",
                    "--data-dir",
                    args.data_dir,
                    *analysis_checkpoint_args,
                    "--output-dir",
                    str(Path(args.results_dir) / "artificial_missingness"),
                    "--tasks",
                    "Task1,Task2,Task3",
                    "--batch-size",
                    "512",
                    "--rule-confidence-threshold",
                    str(args.rule_confidence_threshold),
                    "--model-confidence-threshold",
                    str(args.model_confidence_threshold),
                    "--device",
                    "cpu",
                ],
            ),
            args.dry_run,
        )

    if not args.skip_feature_importance:
        invoke_step(
            "10. Feature-group importance and inference-time ablation",
            new_python_command(
                args,
                [
                    "feature_importance_v74.py",
                    "--data-dir",
                    args.data_dir,
                    *analysis_checkpoint_args,
                    "--output-dir",
                    str(Path(args.results_dir) / "feature_importance"),
                    "--tasks",
                    "Task1,Task2,Task3",
                    "--batch-size",
                    "512",
                    "--permutation-repeats",
                    str(args.feature_importance_repeats),
                    "--device",
                    "cpu",
                ],
            ),
            args.dry_run,
        )

    if not args.skip_model_profile:
        profile_args = [
            "model_profile_v74.py",
            "--skip-default-checkpoint",
            "--checkpoint-glob",
            str(Path(args.results_dir) / "five_runs" / "**" / "best_model.pth"),
            "--hybrid-predictions",
            str(Path(args.paper_dir) / "hybrid_evaluation" / "hybrid_predictions.csv"),
            "--output-dir",
            str(Path(args.results_dir) / "model_profile"),
            "--device",
            "cpu",
            "--batch-sizes",
            "1,32,512",
            "--warmup",
            "3",
            "--repeats",
            "10",
            "--rule-confidence-threshold",
            str(args.rule_confidence_threshold),
        ]
        if args.profile_ml_baselines or not args.skip_profile_ml_baselines:
            profile_args.append("--include-ml-baselines")
        if args.quick_profile_ml_baselines:
            profile_args.append("--quick-ml")
        invoke_step(
            "11. Optional model, hybrid, and deployment/profile estimate",
            new_python_command(
                args,
                profile_args,
            ),
            args.dry_run,
        )

    if args.run_locked_test:
        locked_args = [
            "run_locked_test_v74.py",
            "--data-dir",
            args.data_dir,
            "--results-dir",
            args.locked_results_dir,
            "--paper-dir",
            args.locked_paper_dir,
            "--split-manifest",
            str(Path(args.results_dir) / "split_protocol" / "split_manifest.csv"),
            "--split-output-dir",
            str(Path(args.results_dir) / "split_protocol"),
            "--seeds",
            args.seeds,
            "--config-preset",
            args.config_preset,
            "--run-config-file",
            args.run_config_file,
            "--experiments",
            args.experiments,
            "--single-task-target",
            args.single_task_target,
            "--rule-confidence-threshold",
            str(args.rule_confidence_threshold),
            "--model-confidence-threshold",
            str(args.model_confidence_threshold),
            "--heatmap-mode",
            args.heatmap_mode,
        ]
        if args.locked_allow_overwrite:
            locked_args.append("--allow-overwrite")
        invoke_step(
            "12. Optional formal locked-test pipeline",
            new_python_command(args, locked_args),
            args.dry_run,
        )

    if not args.skip_sync_all:
        include_raw_data = args.include_raw_data or args.package_type == "execution_ready"
        include_checkpoints = args.include_checkpoints or args.package_type == "execution_ready"
        include_large_prediction_rows = args.include_large_prediction_rows or args.package_type == "execution_ready"
        sync_args = [
            "sync_all_outputs_v74.py",
            "--root",
            ".",
            "--all-dir",
            args.all_dir,
            "--clean",
            "--package-type",
            args.package_type,
        ]
        if not include_large_prediction_rows:
            sync_args.append("--skip-large-predictions")
        if include_raw_data:
            sync_args.append("--include-raw-data")
        if include_checkpoints:
            sync_args.append("--include-checkpoints")
        if args.skip_model_profile:
            sync_args.extend(["--exclude-dir", "results_v74/model_profile"])
        if args.skip_feature_importance:
            sync_args.extend(["--exclude-dir", "results_v74/feature_importance"])
        if args.skip_artificial_missingness:
            sync_args.extend(["--exclude-dir", "results_v74/artificial_missingness"])
        if args.skip_calibration:
            sync_args.extend(["--exclude-dir", "results_v74/calibration_analysis"])
            sync_args.extend(["--exclude-dir", "results_v74/calibration_analysis_locked_test"])
        if args.skip_ml_baseline:
            sync_args.extend(["--exclude-dir", "results_v74/ml_baselines"])
        invoke_step(
            "13. Sync formal outputs to all/",
            new_python_command(args, sync_args),
            args.dry_run,
        )

    invoke_local_step("14. Verify key outputs", lambda: verify_key_outputs(args), args.dry_run)

    if args.run_dashboard:
        dashboard_command = ["streamlit", "run", "dashboard_three_tasks_metaIRL_v74.py"]
        invoke_step("15. Launch Streamlit dashboard", dashboard_command, args.dry_run)

    write_section("run_all_v74 completed")
    print("All configured steps finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
