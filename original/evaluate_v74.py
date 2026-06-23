import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str):
    plt.figure(figsize=(8, 4.8))
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_heatmap(cm, out_path: Path, title: str):
    cm = np.array(cm)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results_v74")
    p.add_argument("--paper_dir", type=str, default="paper_v74")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    paper_dir = Path(args.paper_dir)
    figs_dir = paper_dir / "figs"
    tables_dir = paper_dir / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_csv}")

    sdf = pd.read_csv(summary_csv)
    sdf.to_csv(tables_dir / "summary_table.csv", index=False)

    metric_col = "Task1__hearing_degree_WHO_PTAbased_right_macro_f1__mean"
    if metric_col in sdf.columns:
        save_bar(
            sdf[["exp_name", metric_col]].dropna(),
            "exp_name",
            metric_col,
            figs_dir / "ablation_task1_right_macro_f1.png",
            "Ablation on Task1 Right Macro-F1",
        )

    primary_run = results_dir / "full_seed_0" / "eval_summary.json"
    if primary_run.exists():
        obj = json.loads(primary_run.read_text(encoding="utf-8"))
        for task_name, cm_dict in obj.get("confusion_matrices", {}).items():
            for lbl, cm in cm_dict.items():
                safe_lbl = lbl.replace("/", "_")
                save_heatmap(cm, figs_dir / f"cm_{task_name}_{safe_lbl}.png", f"{task_name} - {lbl}")

    manifest = {
        "results_dir": str(results_dir.resolve()),
        "paper_dir": str(paper_dir.resolve()),
        "generated_tables": sorted([p.name for p in tables_dir.glob("*")]),
        "generated_figs": sorted([p.name for p in figs_dir.glob("*")]),
    }
    with open(paper_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[MetaIRL v7.4] Paper assets saved → {paper_dir.resolve()}")


if __name__ == "__main__":
    main()
