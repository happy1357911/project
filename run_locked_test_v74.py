import argparse
import subprocess
import sys
from pathlib import Path


LOCKED_EXPECTED_FILES = [
    Path("tables/summary_locked_test_no_support_all_configs.csv"),
    Path("tables/all_runs_locked_test_no_support_all_configs.csv"),
    Path("tables/per_class_locked_test_no_support_all_configs.csv"),
    Path("hybrid_evaluation/hybrid_locked_test_summary.csv"),
    Path("tables/main_hybrid_summary_locked_test.csv"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def as_cmd_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def run_step(title: str, cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n=== {title} ===")
    print(as_cmd_text(cmd))
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def assert_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def resolved_child(root: Path, child: Path) -> Path:
    path = child if child.is_absolute() else root / child
    return path.resolve()


def ensure_safe_paths(
    root: Path,
    results_dir: Path,
    paper_dir: Path,
    allow_overwrite: bool,
    skip_train: bool,
    skip_evaluate: bool,
) -> None:
    canonical_results = (root / "results_v74").resolve()
    if results_dir.resolve() == canonical_results:
        raise ValueError(
            "Refusing to run locked-test into results_v74. "
            "Use a separate directory such as results_v74_locked_test."
        )

    runs_root = results_dir / "five_runs"
    if runs_root.exists() and not (allow_overwrite or skip_train):
        raise FileExistsError(
            f"{runs_root} already exists. Re-run with --allow-overwrite, "
            "or use --skip-train to reuse existing locked-test training outputs."
        )

    if paper_dir.resolve() in {root.resolve(), results_dir.resolve()}:
        raise ValueError(f"Unsafe paper_dir target: {paper_dir}")
    if paper_dir.exists() and not (allow_overwrite or skip_evaluate):
        raise FileExistsError(
            f"{paper_dir} already exists. Re-run with --allow-overwrite, "
            "or use --skip-evaluate to leave existing paper tables untouched."
        )


def compile_sources(root: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "py_compile",
        "split_protocol_v74.py",
        "train_v74.py",
        "evaluate_v74.py",
        "baseline_rules_v74.py",
        "hybrid_evaluation_v74.py",
    ]
    run_step("Compile locked-test dependencies", cmd, root, dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v7.4 formal locked-test pipeline in an isolated output directory."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--results-dir", default="results_v74_locked_test")
    parser.add_argument("--paper-dir", default="paper_v74_locked_test")
    parser.add_argument("--split-manifest", default="results_v74/split_protocol/split_manifest.csv")
    parser.add_argument("--split-output-dir", default="results_v74/split_protocol")
    parser.add_argument("--locked-test-ratio", type=float, default=0.2)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config-preset", default="all", choices=["recommended", "ablation", "all"])
    parser.add_argument("--experiments", default="full,no_meta,no_irl,single_task")
    parser.add_argument("--single-task-target", default="all", choices=["Task1", "Task2", "Task3", "all"])
    parser.add_argument("--rule-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--heatmap-mode", default="seed0", choices=["none", "seed0", "all"])
    parser.add_argument("--refresh-split-protocol", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-rule-baseline", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = repo_root()
    data_dir = resolved_child(root, Path(args.data_dir))
    results_dir = resolved_child(root, Path(args.results_dir))
    paper_dir = resolved_child(root, Path(args.paper_dir))
    split_manifest = resolved_child(root, Path(args.split_manifest))
    split_output_dir = resolved_child(root, Path(args.split_output_dir))

    ensure_safe_paths(
        root,
        results_dir,
        paper_dir,
        args.allow_overwrite,
        args.skip_train,
        args.skip_evaluate,
    )
    compile_sources(root, args.dry_run)
    if args.compile_only:
        return

    if args.refresh_split_protocol or not split_manifest.exists():
        run_step(
            "Build grouped locked-test split protocol",
            [
                sys.executable,
                "split_protocol_v74.py",
                "--data_dir",
                str(data_dir),
                "--output_dir",
                str(split_output_dir),
                "--seeds",
                args.seeds,
                "--locked_test_ratio",
                str(args.locked_test_ratio),
            ],
            root,
            args.dry_run,
        )

    if not args.dry_run:
        assert_file(split_manifest, "Missing locked split manifest")

    if not args.skip_train:
        run_step(
            "Train models with locked-test manifest",
            [
                sys.executable,
                "train_v74.py",
                "--data_dir",
                str(data_dir),
                "--results_dir",
                str(results_dir),
                "--config_preset",
                args.config_preset,
                "--seeds",
                args.seeds,
                "--experiments",
                args.experiments,
                "--single_task_target",
                args.single_task_target,
                "--locked_split_manifest",
                str(split_manifest),
            ],
            root,
            args.dry_run,
        )

    if not args.skip_evaluate:
        run_step(
            "Export locked-test paper tables",
            [
                sys.executable,
                "evaluate_v74.py",
                "--results_dir",
                str(results_dir),
                "--paper_dir",
                str(paper_dir),
                "--run_mode",
                "multi",
                "--heatmap_mode",
                args.heatmap_mode,
            ],
            root,
            args.dry_run,
        )

    rule_output_dir = results_dir / "rule_baselines_phase3"
    if not args.skip_rule_baseline:
        run_step(
            "Refresh rule baseline predictions for locked-test hybrid merge",
            [
                sys.executable,
                "baseline_rules_v74.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(rule_output_dir),
            ],
            root,
            args.dry_run,
        )

    if not args.skip_hybrid:
        run_step(
            "Build locked-test hybrid summaries",
            [
                sys.executable,
                "hybrid_evaluation_v74.py",
                "--results_dir",
                str(results_dir),
                "--rule_predictions",
                str(rule_output_dir / "rule_baseline_predictions.csv"),
                "--output_dir",
                str(paper_dir / "hybrid_evaluation"),
                "--paper_tables_dir",
                str(paper_dir / "tables"),
                "--mode",
                "locked_test",
                "--rule_confidence_threshold",
                str(args.rule_confidence_threshold),
            ],
            root,
            args.dry_run,
        )

    if not args.dry_run:
        missing = [str(paper_dir / path) for path in LOCKED_EXPECTED_FILES if not (paper_dir / path).exists()]
        if missing:
            raise FileNotFoundError("Missing expected locked-test outputs:\n" + "\n".join(missing))
        print(f"\nLocked-test pipeline completed: {paper_dir}")


if __name__ == "__main__":
    main()
