# 輸出檔案 Label 與欄位意義說明

<!-- 2026-06-20-edge-profile-deferred -->

## 2026-06-20 決策：暫不執行 edge 端估計

目前先不把 edge/deployment latency profile 作為本輪必跑項目。主流程優先保留模型訓練、locked-test、rule/model/hybrid、missingness robustness、calibration 與 baseline 結果；`model_profile_v74.py` 與 `results_v74/model_profile/` 先列為後續需要 edge / IoMT deployment 敘事時再單獨補跑的項目。

<!-- 2026-06-19-codex-update -->

## 2026-06-19 Update: P0/P1 and full pipeline

Status: P0/P1 code changes are implemented and checked against the actual scripts, not only against the suggestion text.

Implemented changes:
- `clinical_rules_v74.py`: added `RuleDecision` for rule label, coverage, confidence, compatible labels, and warning flags.
- `preprocessing_v74.py`, `train_v74.py`, `dashboard_three_tasks_metaIRL_v74.py`, and `error_analysis_v74.py`: aligned rule/reward/dashboard/error-analysis logic with the shared rule decision.
- `train_v74.py`: added training-only structured missingness augmentation and added `run_06_rr_k1_small_missingaug_r010` plus `run_07_rr_k1_small_missingaug_noirl`.
- `artificial_missingness_v74.py`: added structured Task2/Task3 missingness scenarios and model-only/rule-forced/rule-abstain-as-error/hybrid-rule-first strategy outputs.
- `ml_baselines_v74.py`: added `ml_baseline_summary_5seed.csv` and `ml_baseline_per_class_5seed.csv` aliases.
- `model_profile_v74.py`: added `model_profile_summary.csv` for latency, parameters, checkpoint size, and estimated FP32 parameter memory.
- `hybrid_evaluation_v74.py`: separated no-support and locked-test output filenames so locked-test no longer overwrites `main_hybrid_summary.csv`.
- `run_all_v74.py`: reordered the real pipeline as compile -> split protocol -> train with locked manifest -> baselines -> error/evaluate -> no-support hybrid -> locked-test hybrid -> calibration -> artificial missingness -> model profile -> sync -> verify.

Main command:
```powershell
& "C:\Users\ASUS\anaconda3\envs\project2\python.exe" run_all_v74.py --python "C:\Users\ASUS\anaconda3\envs\project2\python.exe"
```

Verification already completed:
- project2 venv full `py_compile` passed.
- `run_all_v74.py --dry-run` showed the corrected order and complete command chain.
- `sync_all_outputs_v74.py` required-file list check returned `missing_files=[]`.

<!-- /2026-06-19-codex-update -->


更新日期：2026-06-12  
適用版本：目前 v7.4 專案、正式輸出與 `all/` 同步狀態

## 1. Label 定義

### Task1：聽損程度

| Label column | 類別 |
|---|---|
| `hearing_degree_WHO_PTAbased` | Normal hearing, Mild hearing loss, Moderate hearing loss, Moderately severe hearing loss, Severe hearing loss, Profound hearing loss, Complete or total hearing loss |

### Task2：聽損種類

| Label column | 類別 |
|---|---|
| `hearing_type` | `WNL`, `SNHL`, `CHL`, `MHL` |

已排除的非標準 label：

- `high tone loss`
- `low tone loss`
- `high and low tone loss`
- 空值或非標準 hearing type

### Task3：Tympanogram type

| Label column | 類別 |
|---|---|
| `tymp_type` | `A`, `B`, `C` |

## 2. Feature 欄位

目前模型實際輸入為 union features，共 49 維。

### Task1 features

| Feature | 意義 |
|---|---|
| `ac_500Hz` | 單耳 AC 500Hz |
| `ac_1000Hz` | 單耳 AC 1000Hz |
| `ac_2000Hz` | 單耳 AC 2000Hz |
| `ac_4000Hz` | 單耳 AC 4000Hz |
| `ac_PTA` | 單耳 PTA |

### Task2 features

Task2 共 32 個 features。

| Feature group | 欄位 |
|---|---|
| AC values | `ac_500Hz`, `ac_1000Hz`, `ac_2000Hz`, `ac_4000Hz` |
| AC NR flags | `ac_500Hz_nr`, `ac_1000Hz_nr`, `ac_2000Hz_nr`, `ac_4000Hz_nr` |
| BC values | `bc_500Hz`, `bc_1000Hz`, `bc_2000Hz`, `bc_4000Hz` |
| BC NR flags | `bc_500Hz_nr`, `bc_1000Hz_nr`, `bc_2000Hz_nr`, `bc_4000Hz_nr` |
| BC missing flags | `bc_500Hz_missing`, `bc_1000Hz_missing`, `bc_2000Hz_missing`, `bc_4000Hz_missing` |
| ABG values | `abg_500Hz`, `abg_1000Hz`, `abg_2000Hz`, `abg_4000Hz` |
| ABG missing flags | `abg_500Hz_missing`, `abg_1000Hz_missing`, `abg_2000Hz_missing`, `abg_4000Hz_missing` |
| ABG censored flags | `abg_500Hz_censored`, `abg_1000Hz_censored`, `abg_2000Hz_censored`, `abg_4000Hz_censored` |

Task2 不使用：

- `ac_mean`
- `bc_mean`
- `abg_mean`
- `PTA` 作為 hearing type rule

### Task3 features

Task3 共 16 個 features。

| Feature group | 欄位 |
|---|---|
| Raw values | `tymp_Vea`, `tymp_peak_daPa`, `tymp_peak_mmho`, `tymp_Width_daPa` |
| real zero flags | `tymp_Vea_real_zero`, `tymp_peak_daPa_real_zero`, `tymp_peak_mmho_real_zero`, `tymp_Width_daPa_real_zero` |
| missing zero flags | `tymp_Vea_missing_zero`, `tymp_peak_daPa_missing_zero`, `tymp_peak_mmho_missing_zero`, `tymp_Width_daPa_missing_zero` |
| NP zero flags | `tymp_Vea_np_zero`, `tymp_peak_daPa_np_zero`, `tymp_peak_mmho_np_zero`, `tymp_Width_daPa_np_zero` |

## 3. 訓練輸出檔案

每個 seed / experiment 會輸出到：

```text
results_v74/five_runs/<run_config>/<exp>_seed_<seed>/
```

| 檔案 | 意義 |
|---|---|
| `best_model.pth` | 最佳 checkpoint。 |
| `training_history.csv` | 每個 epoch 的 train loss、val loss、mean macro-F1、task train steps。 |
| `eval_summary.json` | configured validation 結果。若 meta support 啟用，可能使用 validation support。 |
| `eval_summary_no_support.json` | deployment no-support 結果，固定 `support=None`。 |
| `eval_summary_locked_test_no_support.json` | 若使用 locked split manifest，輸出 locked test no-support 結果。 |
| `per_class_metrics.csv` | configured validation 的 per-class precision/recall/F1/support。 |
| `per_class_metrics_no_support.csv` | no-support validation 的 per-class precision/recall/F1/support。 |
| `per_class_metrics_locked_test_no_support.csv` | locked test no-support per-class metrics。 |
| `prediction_rows.csv` | configured validation 的逐筆預測。 |
| `prediction_rows_no_support.csv` | no-support validation 的逐筆預測。 |
| `prediction_rows_locked_test_no_support.csv` | locked test no-support 逐筆預測。 |

## 4. Validation summary 差異

| 檔案 | 使用情境 |
|---|---|
| `eval_summary.json` | configured validation，用來觀察訓練設定本身的驗證表現。 |
| `eval_summary_no_support.json` | deployment validation，固定 `support=None`，更接近 dashboard 實際推論。 |
| `eval_summary_locked_test_no_support.json` | locked test split 上的 no-support final evaluation。 |

常見欄位：

| 欄位 | 意義 |
|---|---|
| `loss` | validation loss |
| `<label_col>_accuracy` | accuracy |
| `<label_col>_macro_f1` | macro-F1 |
| `<label_col>_balanced_accuracy` | balanced accuracy |
| `<label_col>_auc` | AUC；若類別不足可能為 NaN |
| `per_class` | 每個類別的 precision/recall/F1/support |

## 5. `prediction_rows*.csv`

逐筆預測檔通常包含：

| 欄位 | 意義 |
|---|---|
| `task` | Task1 / Task2 / Task3 |
| `label_col` | label 欄位 |
| `case_id` | 病例 ID，用於追蹤與避免 split leakage |
| `ear_side` | right / left |
| `true_name` | 真實 label |
| `pred_name` | 模型預測 label |
| probability 欄位 | 預測機率或信心 |
| `expert_target` | clinical rule / expert consistency target |
| `expert_confidence` | reward confidence；越低代表 rule evidence 越不確定 |

`prediction_rows_no_support.csv` 固定 `support=None`，是 dashboard/deployment 最應優先對照的逐筆結果。

## 6. Paper export 輸出

正式位置：

```text
paper_v74/tables/
```

目前已產生：

| 檔案 | 意義 |
|---|---|
| `summary_all_configs.csv` | configured validation 跨 run config 彙整。 |
| `all_runs_all_configs.csv` | configured validation 所有 run 詳細資料。 |
| `per_class_all_configs.csv` | configured validation per-class 彙整。 |
| `summary_no_support_all_configs.csv` | no-support 跨 run config 彙整。 |
| `all_runs_no_support_all_configs.csv` | no-support 所有 run 詳細資料。 |
| `per_class_no_support_all_configs.csv` | no-support per-class 彙整。 |
| `history_all_configs.csv` | 所有訓練 history 彙整。 |
| `run_configs.csv` | run config metadata。 |

## 7. Missing-aware rule baseline 輸出

正式位置：

```text
results_v74/rule_baselines_phase3/
```

| 檔案 | 說明 |
|---|---|
| `rule_baseline_summary.csv` | 各 task 的 forced、covered-only、abstain-as-error 指標。 |
| `rule_baseline_summary.json` | 同上，JSON 格式。 |
| `rule_baseline_predictions.csv` | 每筆 ear-level baseline 預測與 evidence 狀態。 |
| `rule_baseline_evidence_summary.csv` | 各 evidence status 的數量與準確率。 |

重要欄位：

| 欄位 | 意義 |
|---|---|
| `forced_pred_label` | 即使缺值也由 rule 硬判出的 label。 |
| `abstain_pred_label` | 若 baseline evidence 不足，顯示 `INSUFFICIENT_EVIDENCE`。 |
| `baseline_covered` | 是否屬於 baseline 有足夠證據可正式判斷。 |
| `evidence_status` | `complete_evidence`、`bc_missing`、`no_bc_data`、`nr_or_censored`、`missing_tymp_data` 等。 |
| `forced_correct` | forced prediction 是否等於人工 label。 |
| `abstain_correct` | abstain-aware prediction 是否等於人工 label。 |

## 8. Classical ML baseline 輸出

正式位置：

```text
results_v74/ml_baselines/
```

| 檔案 | 意義 |
|---|---|
| `ml_baseline_summary.csv` | 各 task / model / seed 的整體 metrics。 |
| `ml_baseline_per_class.csv` | per-class metrics。 |
| `ml_baseline_confusion_matrices.json` | confusion matrix。 |
| `ml_baseline_skipped_models.csv` | optional boosting 缺套件時的 skipped log。 |
| `ml_baseline_manifest.json` | 執行資訊與輸出摘要。 |

## 9. Error analysis 輸出

正式位置：

```text
paper_v74/error_analysis/
```

| 檔案 | 意義 |
|---|---|
| `prediction_rows_all.csv` | 合併所有逐筆預測。 |
| `error_cases.csv` | 錯誤案例。 |
| `subgroup_metrics.csv` | subgroup metrics。 |
| `confusion_pairs.csv` | true/pred confusion pair 統計。 |
| `rule_model_conflicts.csv` | 模型、人工 label、rule/reward 訊號衝突案例。 |
| `rule_model_conflict_summary.csv` | rule-model conflict 彙整。 |
| `task2_clinical_subgroups.csv` | Task2 clinical subgroup 指標。 |
| `task2_confusion_focus.csv` | Task2 confusion pair 與 clinical flags。 |
| `error_analysis_manifest.json` | 分析摘要。 |

## 10. Split protocol 輸出

正式位置：

```text
results_v74/split_protocol/
```

| 檔案 | 意義 |
|---|---|
| `split_manifest.csv` | 每筆資料的 train/val/test assignment。 |
| `split_summary.csv` | 每個 task / seed / split 的分布摘要。 |
| `split_protocol_manifest.json` | split protocol 執行資訊。 |

## 11. Artificial missingness 輸出

正式位置：

```text
results_v74/artificial_missingness/
```

| 檔案 | 意義 |
|---|---|
| `artificial_missingness_summary.csv` | 不同人工缺失 scenario 的整體指標。 |
| `artificial_missingness_predictions.csv` | 各 scenario 的逐筆預測。 |
| `artificial_missingness_manifest.json` | 執行資訊與 checkpoint。 |

注意：`scenario=none` 是全資料 no-support inference，不等同 formal validation split。

## 12. Model profile 輸出

正式位置：

```text
results_v74/model_profile/
```

| 檔案 | 意義 |
|---|---|
| `model_profile.csv` | parameter count、checkpoint size、CPU latency。 |
| `model_profile_manifest.json` | 執行資訊。 |

重要欄位：

| 欄位 | 意義 |
|---|---|
| `parameter_count` | 模型總參數量。 |
| `trainable_parameter_count` | 可訓練參數量。 |
| `checkpoint_size_mb` | checkpoint 檔案大小。 |
| `latency_mean_ms` | 該 batch size 的平均推論時間。 |
| `latency_per_sample_ms` | 平均到每筆 sample 的推論時間。 |

## 13. Hybrid evaluation 輸出

正式位置：

```text
paper_v74/hybrid_evaluation/
```

| 檔案 | 意義 |
|---|---|
| `hybrid_summary.csv` | model_only、rule_forced、rule_abstain_as_error、hybrid_rule_first 比較。 |
| `hybrid_predictions.csv` | 逐筆 hybrid prediction。 |
| `hybrid_manifest.json` | 執行資訊。 |

## 14. Dashboard 輸出

Dashboard 推論固定：

```python
support=None
```

因此 dashboard 結果應優先與：

```text
eval_summary_no_support.json
per_class_metrics_no_support.csv
prediction_rows_no_support.csv
```

對照。dashboard 現在也會顯示：

- rule label
- abstain rule label
- evidence status
- rule confidence
- rule-model conflict
- warning reasons

## 15. `all/` 同步狀態

`all/` 目前已同步主要程式、md、CSV 與正式輸出。同步 manifest：

```text
all/sync_manifest_v74.json
```

注意：根目錄的 `all.zip` 是舊壓縮檔，不代表目前最新版 `all/`。

## 16. 閱讀結果建議順序

1. 先看 `paper_v74/tables/summary_no_support_all_configs.csv`：deployment 情境整體表現。
2. 再看 `paper_v74/tables/per_class_no_support_all_configs.csv`：少數類別是否有學到。
3. 再看 `results_v74/rule_baselines_phase3/rule_baseline_summary.csv`：missing-aware rule baseline。
4. 再看 `results_v74/ml_baselines/ml_baseline_summary.csv`：classical ML 對照。
5. 再看 `paper_v74/error_analysis/subgroup_metrics.csv` 與 `rule_model_conflict_summary.csv`：錯誤來源。
6. 最後看 `results_v74/artificial_missingness/artificial_missingness_summary.csv` 與 `results_v74/model_profile/model_profile.csv`：缺失壓力測試與部署成本。

---

## 2026-06-18 最新輸出欄位更新

新增正式輸出與欄位如下：`paper_v74/tables/main_hybrid_summary.csv` 彙整 rule/model/hybrid 的 Task1/Task2/Task3 macro-F1 與 coverage；`rule_true_model_conflicts.csv` / `rule_true_model_conflict_summary.csv` 記錄 `rule_label`、`true_label`、`model_pred`、`model_correct`、`rule_correct`、`rule_model_agree`、`rule_true_model_conflict_type`；per-class tables 新增 `low_support_flag`、`low_support_reason`、`low_support_threshold`；`calibration_summary.csv`、`calibration_ece_bins.csv`、`confidence_threshold_curve.csv` 分別提供 ECE/Brier/threshold coverage；`deployment_profile.csv` 記錄 neural、hybrid rule-first、classical ML latency rows。

大型中間檔如 `prediction_rows_all.csv` 與 `hybrid_predictions.csv` 屬可重建分析輸出；正式打包時可用 `sync_all_outputs_v74.py --skip-large-predictions` 跳過。

## 2026-06-21 ?? Robustness / Importance / Package ??
????/???????

- `artificial_missingness_summary.csv`??? checkpoint?task?scenario?strategy ? Accuracy?Macro-F1?Balanced Accuracy?Macro Sensitivity?Macro Specificity?
- `artificial_missingness_degradation_summary.csv`??? complete-data scenario ???????????
- `artificial_missingness_per_class.csv`???? TP/FP/FN/TN?Sensitivity?Specificity?Precision?F1 ? support?
- `evidence_compensation_summary.csv`??????????????????????????
- `feature_importance_baseline.csv`?feature importance/ablation ? baseline performance?
- `feature_group_ablation_summary.csv`?? feature group ? mean-mask ????? drop?
- `feature_group_permutation_detail.csv`??? permutation repeat ?????????? all/ ???????????
- `feature_group_permutation_importance.csv`?? feature group ? permutation importance ??????
- `hybrid_summary.csv` / `main_hybrid_summary.csv`??? `hybrid_rule_first_confidence_gate` strategy ? `low_confidence_abstain_rate` ???
- `sync_manifest_v74.json`??? `include_checkpoints`?`checkpoint_files_in_all`?`large_prediction_files_in_all` ? package inventory ???

?????
- `scenario_family`?????????? `bc_missing`?`tymp_missing`?`np_like`?
- `removed_evidence`?? scenario ??????????????
- `fallback_evidence`???/???? scenario ?????????????
- `macro_sensitivity` / `macro_specificity`???? sensitivity/specificity ?? macro average?
- `*_drop_from_complete`??? checkpoint?? task?? strategy ? complete-data baseline ???????
- `low_confidence_abstain_rate`?confidence-gated hybrid ?????????? `INSUFFICIENT_EVIDENCE` ????

