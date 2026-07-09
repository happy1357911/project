<!-- 2026-07-08-directional-abg-rule-gate -->

## 2026-07-08 Current Rule and Hybrid Gate

This block records the current code behavior. If older historical sections mention the old absolute-gap formula, the old 10 dB ABG rule, or direct Task3 B/C hard labels, this block is authoritative.

### Task2 directional ABG

- Task2 rule no longer uses the old absolute-gap formula.
- Meaningful ABG is directional: `AC - BC >= 15 dB`.
- `BC - AC >= 15 dB` is not ABG. It is flagged as `negative_abg_or_measurement_inconsistency` and falls back to the model.
- `10 <= AC - BC < 15 dB` is `abg_borderline`; it does not trigger CHL/MHL and falls back to the model.
- AC 6000/8000 Hz is model input and supplemental warning evidence only. It does not independently decide SNHL in the rule path.

### Task2 rule-first gate

- Forced rule labels may still output WNL/SNHL/CHL/MHL for baseline analysis.
- Official rule-first currently allows only covered `SNHL` and `WNL` cases.
- `MHL` and `CHL` are kept for forced-rule analysis but are blocked from official hybrid rule-first because the current labels and simplified numeric rules still conflict for those classes.
- Missing data, borderline ABG, negative ABG, high-frequency-only abnormality, and unclear rule combinations all set `baseline_covered=False` and fall back to the model.

### Task3 rule-first gate

- `peak NP + compliance NP` remains valid Type B evidence.
- `peak_daPa <= -300` and `-300 < peak_daPa <= -150` are B/C-uncertain and fall back to the model.
- `peak_daPa > -150` can be rule-first Type A only when there is no low-compliance or wide-width warning.
- `compliance < 0.20` or `width >= 200` is A/B-uncertain and falls back to the model.
- Vea missing is an evidence warning only; it does not decide A/B/C by itself.

### Hybrid gate

Official hybrid uses rule only when all conditions hold: valid `rule_decision_label`, `baseline_covered=True`, `complete_for_rule=True`, `rule_confidence >= 0.8`, and no hard warning in `warning_reasons`. Otherwise it uses the model prediction, with low-confidence abstain only when model confidence is below threshold. `rule_forced` is analysis-only and is not the deployed decision policy.

<!-- /2026-07-08-directional-abg-rule-gate -->

<!-- 2026-07-01-inline-run-configs -->

## 2026-07-01 目前參數設定狀態

本專案已取消外部參數檔。正式訓練參數集中在 `train_v74.py` 內維護：

- `MODEL_SIZE_CONFIGS`：base / small / tiny 的模型大小設定。
- `MISSING_AUG_PROFILES`：Task1、Task2、Task3 的缺失資料 augmentation 權重。
- `RUN_CONFIGS`：15 組正式訓練參數。
- `ABLATION_RUN_CONFIGS`：5 組 ablation 參數。

`train_v74.py` 仍使用 `--config_preset recommended|ablation|all` 選擇要跑正式組、ablation 組或全部 20 組。`run_all_v74.py` 與 `run_locked_test_v74.py` 不再接受外部參數檔路徑。若要改參數或分電腦跑，請直接在同一套程式的 `train_v74.py` 內調整參數組合或自行改要執行的 config，不再使用外部分工檔。同步程式也不再同步舊 config 目錄，正式封包中的舊 config 目錄已清除。

<!-- /2026-07-01-inline-run-configs -->
# Cleanup Candidates 2026-06-18

<!-- 2026-06-26-current-cleanup-policy -->

## 2026-06-26 目前清理與 package 判斷規則
目前程式碼對齊重點：
- Task1 資料來源：`task1_all_three_common14_v1.csv`。
- Task2/Task3 資料來源：`task2_3_pure_data(6_24).xlsx`。
- 目前 feature 數：Task1 = 5、Task2 = 36、Task3 = 16、三任務 union = 53。
- Task2 model 輸入包含 AC 500/1000/2000/4000/6000/8000 Hz、六頻 AC NR flags、BC 500/1000/2000/4000 Hz、BC NR/missing flags、ABG 500/1000/2000/4000 Hz 的 value/missing/censored flags。
- Task2 rule 使用 AC 500/1000/2000/4000 Hz 加上 6000/8000 Hz 至少一個高頻存在作為完整性條件；ABG 以 `AC-BC>=15 dB` 判定 clear ABG，`10<=AC-BC<15 dB` 只作 borderline warning。
- Task3 current rule: peak <= -150 is B/C-uncertain and falls back to model; clear A requires peak > -150 without low-compliance or wide-width warning.
- Hybrid rule-first 改以 `rule_confidence` / `rule_evidence_score` 門檻決定是否採用 rule；score 達門檻時採用 rule，score 不足時 fallback model，若 model confidence 低於門檻則可 abstain。
- `train_v74.py 內建 RUN_CONFIGS/ABLATION_RUN_CONFIGS` 為完整 20 組設定：15 組 `run_configs` 加 5 組 `ablation_run_configs`。
- `all/` 預設是 `analysis_only` package，不含 raw data、checkpoint、large row-level prediction；需要可重跑包才使用 `--package-type execution_ready`。
- 本輪 edge/model profile 仍可用 `--skip-model-profile` 延後，不作為目前主流程必跑項。
目前清理資料夾時，請先看 `all/sync_manifest_v74.json`，不要只看資料夾名稱判斷是否缺檔。

| 類別 | 建議處理 |
|---|---|
| `train_v74.py 內建 RUN_CONFIGS/ABLATION_RUN_CONFIGS` | 必須保留，這是完整 20 組訓練設定。 |
| `results_v74/split_protocol/` | 必須保留或可重建；兩台電腦最好使用同一份 `split_manifest.csv`。 |
| `results_v74_part_A/` / `results_v74_part_B/` | 分工訓練完成前必須保留；合併 `five_runs/*` 並確認後，才可考慮清理。 |
| `.pth/.pt/.ckpt` | 預設不進 `analysis_only all/`；若要可重跑 package 才用 `execution_ready` 或 `--include-checkpoints`。 |
| large row-level prediction | 預設不進 `analysis_only all/`；若 GPT 需要逐筆分析再另外包含。 |
| `results_v74/model_profile/` | 本輪 edge/profile 延後時可在 sync 時排除，但原始資料夾是否刪除要先確認是否仍需 latency 敘事。 |

建議清理原則：先確認 `sync_manifest_v74.json` 的 `package_type`、`include_raw_data`、`include_checkpoints`、`include_large_prediction_rows`、`excluded_dirs`，再決定是否刪除或重新同步。

<!-- /2026-06-26-current-cleanup-policy -->

## 2026-07-02 清理狀態補充

本輪沒有新增新的 GPT 建議延伸檔。後續 GPT/教授建議統一記錄於 modification_history.md 與對話紀錄.md；正式論文寫作提醒則應整理在主要說明文件中，避免再次產生多個用途相近的 action-plan Markdown。
