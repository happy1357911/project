# Cleanup Candidates 2026-06-18

<!-- 2026-06-26-current-cleanup-policy -->

## 2026-06-26 目前清理與 package 判斷規則
目前程式碼對齊重點：
- Task1 資料來源：`task1_all_three_common14_v1.csv`。
- Task2/Task3 資料來源：`task2_3_pure_data(6_24).xlsx`。
- 目前 feature 數：Task1 = 5、Task2 = 36、Task3 = 16、三任務 union = 53。
- Task2 model 輸入包含 AC 500/1000/2000/4000/6000/8000 Hz、六頻 AC NR flags、BC 500/1000/2000/4000 Hz、BC NR/missing flags、ABG 500/1000/2000/4000 Hz 的 value/missing/censored flags。
- Task2 rule 使用 AC 500/1000/2000/4000 Hz 加上 6000/8000 Hz 至少一個高頻存在作為完整性條件；ABG 以 `abs(AC-BC)>10 dB` 判定，8-10 dB 只作 borderline warning。
- Task3 rule：`peak_daPa <= -300` 判 B，`-300 < peak_daPa <= -150` 判 C 且 confidence 較低，`peak_daPa > -150` 判 A；NP peak 加 NP compliance 判 B。
- Hybrid rule-first 只有在 `complete_for_rule=True`、`baseline_covered=True` 且 `rule_confidence` 達門檻時採用 rule；否則使用 model，若 model confidence 低於門檻則可 abstain。
- `configs/run_configs_v74.json` 為完整 20 組設定：15 組 `run_configs` 加 5 組 `ablation_run_configs`。
- `configs/splits/run_configs_v74_part_A.json` 與 `configs/splits/run_configs_v74_part_B.json` 各選 10 組，供兩台電腦分工訓練；沒有 A/B 自動 fallback。
- `all/` 預設是 `analysis_only` package，不含 raw data、checkpoint、large row-level prediction；需要可重跑包才使用 `--package-type execution_ready`。
- 本輪 edge/model profile 仍可用 `--skip-model-profile` 延後，不作為目前主流程必跑項。
目前清理資料夾時，請先看 `all/sync_manifest_v74.json`，不要只看資料夾名稱判斷是否缺檔。

| 類別 | 建議處理 |
|---|---|
| `configs/run_configs_v74.json` | 必須保留，這是完整 20 組訓練設定。 |
| `configs/splits/run_configs_v74_part_A.json` / `part_B.json` | 必須保留，這是兩台電腦分工訓練設定。 |
| `results_v74/split_protocol/` | 必須保留或可重建；兩台電腦最好使用同一份 `split_manifest.csv`。 |
| `results_v74_part_A/` / `results_v74_part_B/` | 分工訓練完成前必須保留；合併 `five_runs/*` 並確認後，才可考慮清理。 |
| `.pth/.pt/.ckpt` | 預設不進 `analysis_only all/`；若要可重跑 package 才用 `execution_ready` 或 `--include-checkpoints`。 |
| large row-level prediction | 預設不進 `analysis_only all/`；若 GPT 需要逐筆分析再另外包含。 |
| `results_v74/model_profile/` | 本輪 edge/profile 延後時可在 sync 時排除，但原始資料夾是否刪除要先確認是否仍需 latency 敘事。 |

建議清理原則：先確認 `sync_manifest_v74.json` 的 `package_type`、`include_raw_data`、`include_checkpoints`、`include_large_prediction_rows`、`excluded_dirs`，再決定是否刪除或重新同步。

<!-- /2026-06-26-current-cleanup-policy -->