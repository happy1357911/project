# Cleanup Candidates 2026-06-18

<!-- 2026-06-25-p0-p4-gpt-analysis-update -->

## 2026-06-25 all Package Cleanup Rule

The default `all/` package is now an `analysis_only` package for GPT/professor review. It includes code, documentation, paper tables, figures, and compact outputs. It intentionally excludes raw data, checkpoints, and large row-level prediction rows unless explicitly requested.

Before deleting or judging files, inspect `all/sync_manifest_v74.json`:
- `package_type`
- `include_raw_data`
- `include_checkpoints`
- `include_large_prediction_rows`
- `excluded_dirs`

Use `--package-type execution_ready` only when a large re-runnable package is needed.

<!-- /2026-06-25-p0-p4-gpt-analysis-update -->


????? GPT upload package ? `sync_all_outputs_v74.py --clean --skip-large-predictions` ???

?????
- ???? `.pth/.pt/.ckpt`??????????
- ??????? prediction rows??? summary?tables?manifest ??????
- ? all/ ??????? checkpoint??? `--clean` ????????????

????????? root ???? `results_v74/`?`paper_v74/`?`results_v74_locked_test/`???????? all/ package ??????????
