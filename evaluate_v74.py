import argparse
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_DIR_RE = re.compile(r"^(?P<exp>.+)_seed_(?P<seed>\d+)$")
LOW_SUPPORT_THRESHOLD = 5
LOW_SUPPORT_CLASS_RULES = {
    ("Task2", "CHL"): "Task2 CHL is a known sparse class",
    ("Task3", "B"): "Task3 B is a known sparse class",
    ("Task3", "C"): "Task3 C is a known sparse class",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def parse_seed_dir(path: Path):
    match = SEED_DIR_RE.match(path.name)
    if not match:
        return None
    return match.group("exp"), int(match.group("seed"))


def seed_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and parse_seed_dir(p) is not None
    )


def save_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str):
    plot_df = df[[x, y]].copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 4.8))
    plt.bar(plot_df[x].astype(str), plot_df[y].astype(float))
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_heatmap(cm, out_path: Path, title: str, labels: list[str] | None = None):
    cm = np.array(cm)
    plt.figure(figsize=(5.4, 4.8))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if labels and len(labels) == cm.shape[0] == cm.shape[1]:
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def per_class_rows(eval_summary: dict) -> list[dict]:
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
    return rows


def add_low_support_flags(df: pd.DataFrame, threshold: int = LOW_SUPPORT_THRESHOLD) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    support_col = "support" if "support" in out.columns else "support__mean" if "support__mean" in out.columns else None
    support = pd.to_numeric(out[support_col], errors="coerce") if support_col else pd.Series(np.nan, index=out.index)
    reasons = []
    flags = []
    for idx, row in out.iterrows():
        row_reasons = []
        task = str(row.get("task", ""))
        class_name = str(row.get("class_name", ""))
        if pd.notna(support.loc[idx]) and float(support.loc[idx]) < threshold:
            row_reasons.append(f"support_below_{threshold}")
        known_reason = LOW_SUPPORT_CLASS_RULES.get((task, class_name))
        if known_reason:
            row_reasons.append(known_reason)
        flags.append(bool(row_reasons))
        reasons.append("; ".join(row_reasons))
    out["low_support_flag"] = flags
    out["low_support_reason"] = reasons
    out["low_support_threshold"] = int(threshold)
    return out


def save_per_class_table(eval_summary: dict, out_path: Path):
    rows = per_class_rows(eval_summary)
    if rows:
        add_low_support_flags(pd.DataFrame(rows)).to_csv(out_path, index=False, encoding="utf-8-sig")


def class_labels(eval_summary: dict, task_name: str, label_col: str) -> list[str] | None:
    report = eval_summary.get(task_name, {}).get(f"{label_col}_per_class")
    if isinstance(report, dict):
        return list(report.keys())
    return None


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def reset_paper_dir(paper_dir: Path, results_dir: Path) -> Path:
    paper_resolved = paper_dir.resolve()
    results_resolved = results_dir.resolve()
    cwd_resolved = Path.cwd().resolve()

    blocked_targets = {cwd_resolved, cwd_resolved.parent, results_resolved}
    if paper_resolved in blocked_targets:
        raise ValueError(f"Refusing to clear unsafe paper_dir: {paper_resolved}")
    if _is_relative_to(results_resolved, paper_resolved):
        raise ValueError(f"Refusing to clear paper_dir because it contains results_dir: {paper_resolved}")

    preserve_names = ["error_analysis"]
    preserved = []
    temp_root = None
    if paper_resolved.exists():
        if not paper_resolved.is_dir():
            raise NotADirectoryError(f"paper_dir exists but is not a directory: {paper_resolved}")
        temp_root = Path(tempfile.mkdtemp(prefix="paper_v74_preserve_"))
        for name in preserve_names:
            src = paper_resolved / name
            if src.exists():
                dst = temp_root / name
                shutil.move(str(src), str(dst))
                preserved.append((name, dst))
        shutil.rmtree(paper_resolved)

    paper_resolved.mkdir(parents=True, exist_ok=True)
    for name, src in preserved:
        shutil.move(str(src), str(paper_resolved / name))
    if temp_root is not None and temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    return paper_resolved


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst
    return None


def export_macro_f1_bars(summary_df: pd.DataFrame, figs_dir: Path, suffix: str = ""):
    for col in summary_df.columns:
        if col.endswith("_macro_f1__mean"):
            out_name = f"ablation_{safe_name(col.removesuffix('__mean'))}{suffix}.png"
            save_bar(
                summary_df,
                "exp_name",
                col,
                figs_dir / out_name,
                col.replace("__", " ").replace("_", " "),
            )


def collect_per_class(
    seed_paths: list[Path],
    csv_filename: str = "per_class_metrics.csv",
    summary_filename: str = "eval_summary.json",
) -> pd.DataFrame:
    rows = []
    for seed_dir in seed_paths:
        parsed = parse_seed_dir(seed_dir)
        if parsed is None:
            continue
        exp_name, seed = parsed
        csv_path = seed_dir / csv_filename
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            summary_path = seed_dir / summary_filename
            if not summary_path.exists():
                continue
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
            rows_from_json = per_class_rows(obj.get("eval_summary", {}))
            if not rows_from_json:
                continue
            df = pd.DataFrame(rows_from_json)
        df.insert(0, "seed", seed)
        df.insert(0, "exp_name", exp_name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_per_class_exports(per_class_df: pd.DataFrame, tables_dir: Path, all_filename: str, summary_filename: str):
    if per_class_df.empty:
        return
    per_class_df = add_low_support_flags(per_class_df)
    per_class_df.to_csv(tables_dir / all_filename, index=False, encoding="utf-8-sig")
    metric_cols = [c for c in ["precision", "recall", "f1", "support"] if c in per_class_df.columns]
    per_class_agg = (
        per_class_df
        .groupby(["exp_name", "task", "label_col", "class_name"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_class_agg.columns = [
        "__".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple) else str(col)
        for col in per_class_agg.columns
    ]
    per_class_agg = add_low_support_flags(per_class_agg)
    per_class_agg.to_csv(tables_dir / summary_filename, index=False, encoding="utf-8-sig")


def collect_training_history(seed_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for seed_dir in seed_paths:
        parsed = parse_seed_dir(seed_dir)
        if parsed is None:
            continue
        exp_name, seed = parsed
        csv_path = seed_dir / "training_history.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "seed", seed)
        df.insert(0, "exp_name", exp_name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def export_confusion_matrices(
    seed_paths: list[Path],
    figs_dir: Path,
    heatmap_mode: str,
    summary_filename: str = "eval_summary.json",
    suffix: str = "",
    title_suffix: str = "",
):
    if heatmap_mode == "none":
        return []

    selected = []
    for seed_dir in seed_paths:
        parsed = parse_seed_dir(seed_dir)
        if parsed is None:
            continue
        exp_name, seed = parsed
        if heatmap_mode == "seed0" and seed != 0:
            continue
        summary_path = seed_dir / summary_filename
        if summary_path.exists():
            selected.append((exp_name, seed, summary_path))

    generated = []
    for exp_name, seed, summary_path in selected:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        eval_summary = obj.get("eval_summary", {})
        for task_name, cm_dict in obj.get("confusion_matrices", {}).items():
            for label_col, cm in cm_dict.items():
                labels = class_labels(eval_summary, task_name, label_col)
                out_path = figs_dir / f"cm_{safe_name(exp_name)}_seed{seed}_{task_name}_{safe_name(label_col)}{suffix}.png"
                title = f"{exp_name} seed {seed} - {task_name} {label_col}"
                if title_suffix:
                    title = f"{title} ({title_suffix})"
                save_heatmap(
                    cm,
                    out_path,
                    title,
                    labels=labels,
                )
                generated.append(out_path.name)
    return generated


def export_confusion_matrices_from_prediction_rows(
    prediction_rows_path: Path,
    runs_paper_dir: Path,
    heatmap_mode: str,
    chunksize: int = 200_000,
):
    if heatmap_mode == "none" or not prediction_rows_path.exists():
        return []

    mode_suffix = {
        "configured_validation": ("", "configured"),
        "deployment_no_support": ("_no_support", "no-support"),
        "locked_test_no_support": ("_locked_test_no_support", "locked-test no-support"),
    }
    usecols = [
        "config_name",
        "exp_name",
        "seed",
        "evaluation_mode",
        "task",
        "label_col",
        "true_label",
        "pred_label",
    ]
    counts = defaultdict(lambda: defaultdict(int))
    labels_by_key = defaultdict(set)

    for chunk in pd.read_csv(
        prediction_rows_path,
        usecols=usecols,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        chunk = chunk[chunk["evaluation_mode"].isin(mode_suffix)]
        if chunk.empty:
            continue
        chunk["seed"] = pd.to_numeric(chunk["seed"], errors="coerce").astype("Int64")
        if heatmap_mode == "seed0":
            chunk = chunk[chunk["seed"] == 0]
        chunk = chunk.dropna(subset=[
            "config_name",
            "exp_name",
            "seed",
            "evaluation_mode",
            "task",
            "label_col",
            "true_label",
            "pred_label",
        ])
        if chunk.empty:
            continue

        grouped = (
            chunk
            .groupby([
                "config_name",
                "exp_name",
                "seed",
                "evaluation_mode",
                "task",
                "label_col",
                "true_label",
                "pred_label",
            ], dropna=False)
            .size()
        )
        for idx, n in grouped.items():
            config_name, exp_name, seed, evaluation_mode, task_name, label_col, true_label, pred_label = idx
            key = (
                str(config_name),
                str(exp_name),
                int(seed),
                str(evaluation_mode),
                str(task_name),
                str(label_col),
            )
            true_label = str(true_label)
            pred_label = str(pred_label)
            counts[key][(true_label, pred_label)] += int(n)
            labels_by_key[key].update([true_label, pred_label])

    generated = []
    for key, pair_counts in counts.items():
        config_name, exp_name, seed, evaluation_mode, task_name, label_col = key
        suffix, title_suffix = mode_suffix[evaluation_mode]
        labels = sorted(labels_by_key[key])
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for (true_label, pred_label), n in pair_counts.items():
            cm[label_to_idx[true_label], label_to_idx[pred_label]] += n

        figs_dir = runs_paper_dir / safe_name(config_name) / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        out_path = figs_dir / f"cm_{safe_name(exp_name)}_seed{seed}_{task_name}_{safe_name(label_col)}{suffix}.png"
        save_heatmap(
            cm,
            out_path,
            f"{exp_name} seed {seed} - {task_name} {label_col} ({title_suffix})",
            labels=labels,
        )
        generated.append(str(out_path.relative_to(runs_paper_dir)))
    return generated


def refresh_run_fig_manifests(paper_dir: Path):
    for run_dir in sorted(p for p in paper_dir.iterdir() if p.is_dir()):
        figs_dir = run_dir / "figs"
        if not figs_dir.exists():
            continue
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        else:
            manifest = {
                "paper_dir": str(run_dir.resolve()),
            }
        manifest["generated_figs"] = sorted(p.name for p in figs_dir.glob("*"))
        manifest["generated_heatmaps"] = sorted(p.name for p in figs_dir.glob("cm_*.png"))
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def export_one_result_dir(results_dir: Path, paper_dir: Path, heatmap_mode: str = "seed0"):
    figs_dir = paper_dir / "figs"
    tables_dir = paper_dir / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_csv}")

    summary_df = pd.read_csv(summary_csv)
    summary_df.to_csv(tables_dir / "summary_table.csv", index=False, encoding="utf-8-sig")
    copy_if_exists(results_dir / "summary.json", tables_dir / "summary.json")
    copy_if_exists(results_dir / "all_runs_metrics.csv", tables_dir / "all_runs_metrics.csv")
    copy_if_exists(results_dir / "run_config.csv", tables_dir / "run_config.csv")
    copy_if_exists(results_dir / "run_manifest.json", tables_dir / "run_manifest.json")
    export_macro_f1_bars(summary_df, figs_dir)

    summary_no_support_df = pd.DataFrame()
    summary_no_support_csv = results_dir / "summary_no_support.csv"
    if summary_no_support_csv.exists():
        summary_no_support_df = pd.read_csv(summary_no_support_csv)
        summary_no_support_df.to_csv(
            tables_dir / "summary_table_no_support.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copy_if_exists(results_dir / "summary_no_support.json", tables_dir / "summary_no_support.json")
        copy_if_exists(
            results_dir / "all_runs_metrics_no_support.csv",
            tables_dir / "all_runs_metrics_no_support.csv",
        )
        export_macro_f1_bars(summary_no_support_df, figs_dir, suffix="_no_support")

    summary_locked_test_no_support_df = pd.DataFrame()
    summary_locked_test_no_support_csv = results_dir / "summary_locked_test_no_support.csv"
    if summary_locked_test_no_support_csv.exists():
        summary_locked_test_no_support_df = pd.read_csv(summary_locked_test_no_support_csv)
        summary_locked_test_no_support_df.to_csv(
            tables_dir / "summary_table_locked_test_no_support.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copy_if_exists(
            results_dir / "summary_locked_test_no_support.json",
            tables_dir / "summary_locked_test_no_support.json",
        )
        copy_if_exists(
            results_dir / "all_runs_metrics_locked_test_no_support.csv",
            tables_dir / "all_runs_metrics_locked_test_no_support.csv",
        )

    seeds = seed_dirs(results_dir)

    per_class_df = collect_per_class(seeds)
    save_per_class_exports(
        per_class_df,
        tables_dir,
        "per_class_metrics_all.csv",
        "per_class_metrics_summary.csv",
    )

    per_class_no_support_df = collect_per_class(
        seeds,
        csv_filename="per_class_metrics_no_support.csv",
        summary_filename="eval_summary_no_support.json",
    )
    save_per_class_exports(
        per_class_no_support_df,
        tables_dir,
        "per_class_metrics_no_support_all.csv",
        "per_class_metrics_no_support_summary.csv",
    )

    per_class_locked_test_no_support_df = collect_per_class(
        seeds,
        csv_filename="per_class_metrics_locked_test_no_support.csv",
        summary_filename="eval_summary_locked_test_no_support.json",
    )
    save_per_class_exports(
        per_class_locked_test_no_support_df,
        tables_dir,
        "per_class_metrics_locked_test_no_support_all.csv",
        "per_class_metrics_locked_test_no_support_summary.csv",
    )

    history_df = collect_training_history(seeds)
    if not history_df.empty:
        history_df.to_csv(tables_dir / "training_history_all.csv", index=False, encoding="utf-8-sig")

    generated_heatmaps = []
    generated_heatmaps.extend(export_confusion_matrices(seeds, figs_dir, heatmap_mode))
    if not summary_no_support_df.empty:
        generated_heatmaps.extend(
            export_confusion_matrices(
                seeds,
                figs_dir,
                heatmap_mode,
                summary_filename="eval_summary_no_support.json",
                suffix="_no_support",
                title_suffix="no-support",
            )
        )
    if not summary_locked_test_no_support_df.empty:
        generated_heatmaps.extend(
            export_confusion_matrices(
                seeds,
                figs_dir,
                heatmap_mode,
                summary_filename="eval_summary_locked_test_no_support.json",
                suffix="_locked_test_no_support",
                title_suffix="locked-test no-support",
            )
        )

    manifest = {
        "results_dir": str(results_dir.resolve()),
        "paper_dir": str(paper_dir.resolve()),
        "seed_dirs": [p.name for p in seeds],
        "heatmap_mode": heatmap_mode,
        "generated_tables": sorted([p.name for p in tables_dir.glob("*")]),
        "generated_figs": sorted([p.name for p in figs_dir.glob("*")]),
        "generated_heatmaps": sorted(generated_heatmaps),
    }
    with open(paper_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {
        "summary": summary_df.assign(config_name=results_dir.name),
        "summary_no_support": (
            summary_no_support_df.assign(config_name=results_dir.name)
            if not summary_no_support_df.empty else pd.DataFrame()
        ),
        "summary_locked_test_no_support": (
            summary_locked_test_no_support_df.assign(config_name=results_dir.name)
            if not summary_locked_test_no_support_df.empty else pd.DataFrame()
        ),
        "all_runs": (
            pd.read_csv(results_dir / "all_runs_metrics.csv").assign(config_name=results_dir.name)
            if (results_dir / "all_runs_metrics.csv").exists()
            else pd.DataFrame()
        ),
        "all_runs_no_support": (
            pd.read_csv(results_dir / "all_runs_metrics_no_support.csv").assign(config_name=results_dir.name)
            if (results_dir / "all_runs_metrics_no_support.csv").exists()
            else pd.DataFrame()
        ),
        "all_runs_locked_test_no_support": (
            pd.read_csv(results_dir / "all_runs_metrics_locked_test_no_support.csv").assign(config_name=results_dir.name)
            if (results_dir / "all_runs_metrics_locked_test_no_support.csv").exists()
            else pd.DataFrame()
        ),
        "per_class": per_class_df.assign(config_name=results_dir.name) if not per_class_df.empty else pd.DataFrame(),
        "per_class_no_support": (
            per_class_no_support_df.assign(config_name=results_dir.name)
            if not per_class_no_support_df.empty else pd.DataFrame()
        ),
        "per_class_locked_test_no_support": (
            per_class_locked_test_no_support_df.assign(config_name=results_dir.name)
            if not per_class_locked_test_no_support_df.empty else pd.DataFrame()
        ),
        "history": history_df.assign(config_name=results_dir.name) if not history_df.empty else pd.DataFrame(),
    }


def bootstrap_mean_ci(values: pd.Series, *, n_bootstrap: int = 500, seed: int = 74) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boot_means = arr[sample_idx].mean(axis=1)
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def build_statistical_summary(collected: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    source_map = {
        "all_runs": "configured",
        "all_runs_no_support": "no_support",
        "all_runs_locked_test_no_support": "locked_test_no_support",
    }
    metric_cols = ["accuracy", "macro_f1", "balanced_accuracy"]
    wide_metric_map = {
        "acc": "accuracy",
        "macro_f1": "macro_f1",
        "balanced_acc": "balanced_accuracy",
    }
    wide_metric_re = re.compile(
        r"^(?P<task>Task\d+)__(?P<label_col>.+?)_(?P<metric>balanced_acc|macro_f1|acc)$"
    )
    rows = []

    def add_summary_row(base: dict, evaluation_scope: str, metric: str, values: pd.Series):
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return
        n = int(len(values))
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 0 else np.nan
        ci95_half_width = float(1.96 * sem) if np.isfinite(sem) else np.nan
        mean_value = float(values.mean())
        ci_low = mean_value - ci95_half_width if np.isfinite(ci95_half_width) else np.nan
        ci_high = mean_value + ci95_half_width if np.isfinite(ci95_half_width) else np.nan
        bootstrap_low, bootstrap_high = bootstrap_mean_ci(values, seed=74 + len(rows))
        rows.append({
            **base,
            "evaluation_scope": evaluation_scope,
            "metric": metric,
            "n": n,
            "mean": mean_value,
            "std": std,
            "sem": sem,
            "ci95_low": float(np.clip(ci_low, 0.0, 1.0)) if np.isfinite(ci_low) else np.nan,
            "ci95_high": float(np.clip(ci_high, 0.0, 1.0)) if np.isfinite(ci_high) else np.nan,
            "ci95_half_width": ci95_half_width,
            "bootstrap_ci95_low": float(np.clip(bootstrap_low, 0.0, 1.0)) if np.isfinite(bootstrap_low) else np.nan,
            "bootstrap_ci95_high": float(np.clip(bootstrap_high, 0.0, 1.0)) if np.isfinite(bootstrap_high) else np.nan,
            "bootstrap_n": 500,
        })

    for source_key, evaluation_scope in source_map.items():
        frames = collected.get(source_key) or []
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)

        long_group_cols = [
            col for col in ["config_name", "exp_name", "task", "label_col"]
            if col in df.columns
        ]
        if {"task", "label_col"}.issubset(df.columns):
            for key, sub in df.groupby(long_group_cols, dropna=False):
                key_tuple = key if isinstance(key, tuple) else (key,)
                base = dict(zip(long_group_cols, key_tuple))
                for metric in metric_cols:
                    if metric in sub.columns:
                        add_summary_row(base, evaluation_scope, metric, sub[metric])

        wide_metric_cols = []
        for col in df.columns:
            match = wide_metric_re.match(str(col))
            if match:
                wide_metric_cols.append((col, match.groupdict()))
        if not wide_metric_cols:
            continue

        wide_group_cols = [col for col in ["config_name", "exp_name"] if col in df.columns]
        grouped = df.groupby(wide_group_cols, dropna=False) if wide_group_cols else [((), df)]
        for key, sub in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_base = dict(zip(wide_group_cols, key_tuple))
            for col, parsed in wide_metric_cols:
                base = {
                    **group_base,
                    "task": parsed["task"],
                    "label_col": parsed["label_col"],
                }
                add_summary_row(
                    base,
                    evaluation_scope,
                    wide_metric_map[parsed["metric"]],
                    sub[col],
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in ["evaluation_scope", "config_name", "exp_name", "task", "label_col", "metric"] if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)

def export_multi_result_dir(results_dir: Path, paper_dir: Path, heatmap_mode: str):
    runs_root = results_dir / "five_runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Missing multi-run directory: {runs_root}")

    root_tables = paper_dir / "tables"
    runs_paper_dir = paper_dir / "five_runs"
    root_tables.mkdir(parents=True, exist_ok=True)
    runs_paper_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(runs_root / "run_configs.csv", root_tables / "run_configs.csv")

    collected = {
        "summary": [],
        "summary_no_support": [],
        "summary_locked_test_no_support": [],
        "all_runs": [],
        "all_runs_no_support": [],
        "all_runs_locked_test_no_support": [],
        "per_class": [],
        "per_class_no_support": [],
        "per_class_locked_test_no_support": [],
        "history": [],
    }
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        try:
            exported = export_one_result_dir(run_dir, runs_paper_dir / run_dir.name, heatmap_mode=heatmap_mode)
        except FileNotFoundError as exc:
            print(f"[WARN] Skip incomplete result dir: {exc}")
            continue
        for key, df in exported.items():
            if not df.empty:
                collected[key].append(df)

    for key, frames in collected.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                root_tables / f"{key}_all_configs.csv",
                index=False,
                encoding="utf-8-sig",
            )

    statistical_summary = build_statistical_summary(collected)
    if not statistical_summary.empty:
        statistical_summary.to_csv(
            root_tables / "statistical_summary_all_configs.csv",
            index=False,
            encoding="utf-8-sig",
        )

    generated_prediction_heatmaps = export_confusion_matrices_from_prediction_rows(
        paper_dir / "error_analysis" / "prediction_rows_all.csv",
        runs_paper_dir,
        heatmap_mode,
    )
    if generated_prediction_heatmaps:
        refresh_run_fig_manifests(runs_paper_dir)

    manifest = {
        "results_dir": str(results_dir.resolve()),
        "runs_root": str(runs_root.resolve()),
        "paper_dir": str(paper_dir.resolve()),
        "runs_output_dir": str(runs_paper_dir.resolve()),
        "run_dirs": sorted(p.name for p in runs_root.iterdir() if p.is_dir()),
        "run_paper_dirs": sorted(p.name for p in runs_paper_dir.iterdir() if p.is_dir()),
        "generated_tables": sorted([p.name for p in root_tables.glob("*")]),
        "generated_prediction_heatmaps": sorted(generated_prediction_heatmaps),
    }
    with open(paper_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results_v74")
    p.add_argument("--paper_dir", type=str, default="paper_v74")
    p.add_argument("--run_mode", type=str, default="auto", choices=["auto", "single", "multi"])
    p.add_argument("--heatmap_mode", type=str, default="seed0", choices=["none", "seed0", "all"])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    paper_dir = reset_paper_dir(Path(args.paper_dir), results_dir)

    runs_root = results_dir / "five_runs"
    is_multi = runs_root.exists() if args.run_mode == "auto" else args.run_mode == "multi"
    if is_multi:
        export_multi_result_dir(results_dir, paper_dir, heatmap_mode=args.heatmap_mode)
        print(f"[MetaIRL v7.4] Paper assets saved to {paper_dir.resolve()}")
    else:
        export_one_result_dir(results_dir, paper_dir, heatmap_mode=args.heatmap_mode)
        print(f"[MetaIRL v7.4] Paper assets saved to {paper_dir.resolve()}")


if __name__ == "__main__":
    main()
