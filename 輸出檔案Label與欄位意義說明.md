<!-- 2026-07-08-hard-warning-taxonomy-update -->
## 2026-07-08 New output labels

New or newly required outputs:

| File | Meaning |
| --- | --- |
| `paper_v74/error_analysis/task2_rule_label_conflict_audit.csv` | Case-level Task2 audit comparing true label, model prediction, forced rule label, rule coverage, rule confidence, warning flags, and conflict category. |
| `results_v74/artificial_missingness/clinical_missingness_taxonomy.csv` | Scenario-level taxonomy describing what evidence was removed, what fallback evidence remains, the clinical missingness type, and the expected decision policy. |
| `paper_v74/tables/clinical_missingness_taxonomy.csv` | Paper-ready copy of the clinical missingness taxonomy. |

Important fields:

| Field | Meaning |
| --- | --- |
| `clinical_missingness_type` | Normalized category of the missing or degraded evidence, such as `core_bc_missing`, `borderline_diagnostic_evidence`, or `all_tympanogram_missing`. |
| `expected_decision_policy` | Intended rule/model behavior for the scenario, for example model fallback when core evidence is unavailable. |
| `scenario_forced_model_fallback` | Whether the scenario is intentionally forced to model fallback before hybrid rule-first can use a rule. |
| `rule_first_allowed_by_scenario` | Whether the scenario itself permits official rule-first use. |
| `has_negative_abg_warning` | Task2 flag for directional ABG inconsistency where BC is worse than AC by a large margin. |
| `has_abg_borderline_warning` | Task2 flag for borderline ABG evidence that should not be treated as clear ABG. |
| `has_class_gate_warning` | Task2 flag showing that a forced rule class is blocked from official rule-first use. |
| `has_missing_evidence_warning` | Task2 flag for missing or incomplete rule evidence. |

<!-- /2026-07-08-hard-warning-taxonomy-update -->

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

<!-- 2026-07-05-device-routing-update -->

## 2026-07-05 裝置參數與輸出 manifest 欄位更新

目前 `run_all_v74.py` 的神經網路大量運算不再寫死 CPU：

| 腳本 | 新增或更新欄位 | 意義 |
|---|---|---|
| `artificial_missingness_v74.py` | `requested_device` | 使用者要求的裝置，例如 `auto`、`cpu`、`cuda`。 |
| `artificial_missingness_v74.py` | `resolved_device` | 程式實際使用的裝置，例如 `cuda` 或 `cpu`。 |
| `feature_importance_v74.py` | `requested_device` | 使用者要求的裝置。 |
| `feature_importance_v74.py` | `resolved_device` | 程式實際使用的裝置。 |
| `model_profile_v74.py` | `requested_device` | 使用者要求的 profile 裝置。 |
| `model_profile_v74.py` | `resolved_device` | neural checkpoint profile 實際使用的裝置；若只跑 hybrid/ML profile，可能等同 requested value。 |

輸出檔案的研究意義沒有改變：Step 9 仍是人工缺失 robustness stress test，Step 10 仍是 inference-time feature group ablation / permutation importance。這次只改執行裝置與 manifest 可追溯性，不改 metric 定義、欄位語意或輸出資料夾結構。

建議 GPT 分析封包仍可使用：

```powershell
python sync_all_outputs_v74.py --root . --all-dir all --clean --package-type analysis_only --skip-large-predictions --exclude-dir results_v74/model_profile
```

<!-- /2026-07-05-device-routing-update -->
<!-- 2026-07-01-inline-run-configs -->

## 2026-07-01 目前參數設定狀態

本專案已取消外部參數檔。正式訓練參數集中在 `train_v74.py` 內維護：

- `MODEL_SIZE_CONFIGS`：base / small / tiny 的模型大小設定。
- `MISSING_AUG_PROFILES`：Task1、Task2、Task3 的缺失資料 augmentation 權重。
- `RUN_CONFIGS`：15 組正式訓練參數。
- `ABLATION_RUN_CONFIGS`：5 組 ablation 參數。

`train_v74.py` 仍使用 `--config_preset recommended|ablation|all` 選擇要跑正式組、ablation 組或全部 20 組。`run_all_v74.py` 與 `run_locked_test_v74.py` 不再接受外部參數檔路徑。若要改參數或分電腦跑，請直接在同一套程式的 `train_v74.py` 內調整參數組合或自行改要執行的 config，不再使用外部分工檔。同步程式也不再同步舊 config 目錄，正式封包中的舊 config 目錄已清除。

<!-- /2026-07-01-inline-run-configs -->
# 輸出檔案 Label 與欄位意義說明

<!-- 2026-06-27-narrative-alignment -->

## 2026-06-27 文字敘事與命名收斂

本段為目前對外說明的優先採信版本。後續給 GPT、教授或論文草稿分析時，請先以本段理解研究定位；較早日期段落若仍出現舊名稱或舊數字，只作歷史紀錄，不作為目前主張。

目前研究主軸不是單純的神經網路架構或模型準確率比較，而是：

```text
Missing-aware clinical-rule-guided hybrid decision-support framework
for ear-level hearing classification under incomplete audiological evidence.
```

中文定位：聽力師規則引導、缺值感知的混合式聽力輔助判讀系統。

目前敘事原則：
- `MetaIRL`、Transformer、prototype/meta-learning 只作為神經網路元件或消融模組，不作為唯一主貢獻。
- 主要臨床問題是 Task2/Task3 在缺失、NR/截尾值、ABG 邊界值、無 peak / 無 tympanogram 等不完整證據下的可靠輔助判讀。
- 混合式決策的價值是規則優先、模型回退、警示、暫不判讀與規則-模型衝突透明化；不可寫成混合式決策已全面高於規則決策。
- 傳統機器學習基準很強；若實驗協定不同，只能說是具競爭力的 grouped validation 基準，不能直接與 locked-test 神經網路結果作公平勝負結論。
- 人工缺失分析目前是強健性壓力測試證據；除非對齊主要設定與 5 seeds，否則不要寫成最終主要強健性結果。
- IoMT / 部署目前應寫成概念性臨床決策輔助流程；邊緣端 latency/profile 不是本輪主軸，device-shift 主張需等有 device-level 驗證後再放入。
- `all/` 是給 GPT/教授分析的封包；可包含目前使用的原始資料，但沒有 checkpoint 與大型逐列預測時，仍不是完整可重跑的 execution-ready 封包。

<!-- /2026-06-27-narrative-alignment -->

<!-- 2026-06-26-目前-輸出-field-alignment -->

## 2026-06-26 目前輸出與欄位對齊

本段依目前程式碼更新；舊段落中的 特徵 數或輸出名稱若不同，以本段為準。
目前程式碼對齊重點：
- Task1 資料來源：`task1_all_three_common14_v1.csv`。
- Task2/Task3 資料來源：`task2_3_pure_data(6_24).xlsx`。
- 目前特徵數：Task1 = 5、Task2 = 36、Task3 = 16、三任務 union = 53。
- Task2 模型輸入包含 AC 500/1000/2000/4000/6000/8000 Hz、六頻 AC NR 標記、BC 500/1000/2000/4000 Hz、BC NR/缺失標記、ABG 500/1000/2000/4000 Hz 的 數值/缺失/截尾 標記。
- Task2 規則使用 AC 500/1000/2000/4000 Hz 加上 6000/8000 Hz 至少一個高頻存在作為完整性條件；ABG 以 `AC-BC>=15 dB` 判定 clear ABG，`10<=AC-BC<15 dB` 作邊界警示。
- Task3 current rule: peak <= -150 is B/C-uncertain and falls back to model; clear A requires peak > -150 without low-compliance or wide-width warning.
- 混合式決策 規則優先 改以 `rule_confidence` / `rule_evidence_score` 達門檻且 rule label 存在時採用規則；否則使用模型，若模型信心度低於門檻則可暫不判讀。
- `train_v74.py 內建 RUN_CONFIGS/ABLATION_RUN_CONFIGS` 為完整 20 組設定：15 組 `run_configs` 加 5 組 `ablation_run_configs`。
- `all/` 預設是給 GPT/教授分析的 `analysis_only` 封包；可用 `--include-raw-data` 納入目前使用的原始資料，但未包含 checkpoint 與大型逐列預測 時仍不是完整可重跑 封包。
- 本輪 邊緣端/model profile 仍可用 `--skip-model-profile` 延後，不作為目前主流程必跑項。
新增或目前重要欄位：

| 欄位 / 輸出 | 意義 |
|---|---|
| `complete_for_rule` | rule 證據 是否完整；混合式決策 規則優先 必須為 True 才能採 rule。 |
| `baseline_covered` | rule 是否可產生有效 規則 label；邊界 或缺失情境可能為 False。 |
| `rule_confidence` | 規則決策 信心度；Task2 完整證據 通常為 1.0，邊界 會降低；Task3 C 區間為較低 信心度。 |
| `model_confidence` / `confidence` | 模型預測 信心度，用於 校準 與 低信心度 暫不判讀。 |
| `hybrid_decision_reason` | 混合式決策 採 rule、回退 model、規則暫不判讀、低信心 暫不判讀 的原因。 |
| `hybrid_warning_reasons` | 混合式決策的 警示 標記，例如 incomplete rule 資料、low 模型信心度。 |
| `decision_safety_summary.csv` | 彙整低信心、規則暫不判讀、規則/模型 衝突、缺失/NR/NP 等安全性指標。 |
| `statistical_summary_all_configs.csv` | repeated-seed mean/std/SEM/95% CI 摘要。 |
| `feature_group_ablation_summary.csv` | inference-time mean masking ablation 摘要。 |
| `feature_group_permutation_importance.csv` | 特徵群組 permutation importance 摘要。 |
| `sync_manifest_v74.json` | 記錄 封包類型、是否包含 原始資料/checkpoint/大型 逐筆預測列。 |

兩台分工訓練輸出：A/B 電腦可分別使用 `results_v74_part_A/`、`results_v74_part_B/`，最終需把兩邊的 `five_runs/*` 合併回主電腦 `results_v74/five_runs/` 後再跑後處理與同步。

<!-- /2026-06-26-目前-輸出-field-alignment -->

<!-- 2026-06-25-p0-p4-gpt-analysis-update -->

## 2026-06-25 P0-P4 GPT 分析回應更新

本次更新是在核對目前程式碼、最新版 GPT all.zip 分析與教授回饋後完成。目標是讓專案從只看模型分數的敘事，收斂為缺值感知、臨床規則引導、可追溯 locked-test 的決策輔助流程。

已完成更新：
- `sync_all_outputs_v74.py` 現在會標示封包類型：`analysis_only`、`execution_ready` 或 `custom`。
- sync manifest 會明確記錄是否包含原始資料、checkpoint 與大型逐列預測檔案。
- `run_all_v74.py` 會把封包旗標傳給同步流程，並驗證 `sync_manifest_v74.json` 是否符合執行參數。
- `run_locked_test_v74.py` 現在會產生 locked-test error analysis、三方衝突表、校準摘要、ECE bins、Brier score 與信心度門檻 curves。
- `run_all_v74.py` 現在會建立 `results_v74/calibration_analysis_locked_test/`，並用 `--mode all` 執行 error analysis。
- 混合式決策輸出現在包含 `hybrid_decision_reason`、`hybrid_warning_reasons` 與決策原因摘要 CSV。
- 人工缺失現在會針對 incomplete 或 corrupted stress scenario 採用較保守的 scenario-level 模型回退。
- Task2 artificial ABG 邊界 stress 現在使用 10 dB 而不是 15 dB，使其代表邊界而非明確 ABG。
- Error analysis 現在會輸出 `decision_safety_summary.csv`。
- Dashboard 現在會顯示模型信心度、混合式決策原因與低信心度暫不判讀。
- `evaluate_v74.py` 現在會輸出 `statistical_summary_all_configs.csv`，包含 mean、std、SEM 與 clipped 95% CI。

建議正式刷新指令：

```powershell
python run_all_v74.py --skip-model-profile --run-locked-test --locked-allow-overwrite --package-type analysis_only
```

只有在需要可重跑的大型封包時才使用：

```powershell
python run_all_v74.py --skip-model-profile --run-locked-test --locked-allow-overwrite --package-type execution_ready
```

<!-- /2026-06-25-p0-p4-gpt-analysis-update -->


<!-- 2026-06-20-邊緣端-profile-deferred -->

## 2026-06-20 決策：暫不執行 邊緣端估計

目前先不把 邊緣端/部署 latency profile 作為本輪主軸。主流程優先保留模型訓練、locked-test、規則/模型/混合式決策、缺失狀態 強健性、校準、特徵重要性 與 基準 結果；`model_profile_v74.py` 與 `results_v74/model_profile/` 先列為後續需要補強 IoMT 可部署性敘事時再單獨補跑的項目。

<!-- 2026-06-19-codex-update -->

## 2026-06-19 完整流程更新

`run_all_v74.py` 現在是主要完整流程入口。它會先建立 grouped locked-test split，再把 `--locked_split_manifest` 傳給 `train_v74.py`，因此主流程可以同時產生 no-support 與 locked-test no-support 輸出。

使用 project2 環境執行：
```powershell
& "C:\Users\ASUS\anaconda3\envs\project2\python.exe" run_all_v74.py --python "C:\Users\ASUS\anaconda3\envs\project2\python.exe"
```

主要預期輸出包含 `main_hybrid_summary.csv`、`main_hybrid_summary_locked_test.csv`、`three_way_conflict_summary.csv`、`ml_baseline_summary_5seed.csv`、`artificial_missingness_summary.csv` 與 `model_profile_summary.csv`。

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

## 2. 特徵 欄位

目前模型實際輸入為 union 特徵，共 53 維。

### Task1 特徵

| 特徵 | 意義 |
|---|---|
| `ac_500Hz` | 單耳 AC 500Hz |
| `ac_1000Hz` | 單耳 AC 1000Hz |
| `ac_2000Hz` | 單耳 AC 2000Hz |
| `ac_4000Hz` | 單耳 AC 4000Hz |
| `ac_PTA` | 單耳 PTA |

### Task2 特徵

Task2 共 36 個 特徵。

| 特徵 group | 欄位 |
|---|---|
| AC 數值 | `ac_500Hz`, `ac_1000Hz`, `ac_2000Hz`, `ac_4000Hz`, `ac_6000Hz`, `ac_8000Hz` |
| AC NR 標記 | `ac_500Hz_nr`, `ac_1000Hz_nr`, `ac_2000Hz_nr`, `ac_4000Hz_nr`, `ac_6000Hz_nr`, `ac_8000Hz_nr` |
| BC 數值 | `bc_500Hz`, `bc_1000Hz`, `bc_2000Hz`, `bc_4000Hz` |
| BC NR 標記 | `bc_500Hz_nr`, `bc_1000Hz_nr`, `bc_2000Hz_nr`, `bc_4000Hz_nr` |
| BC 缺失標記 | `bc_500Hz_missing`, `bc_1000Hz_missing`, `bc_2000Hz_missing`, `bc_4000Hz_missing` |
| ABG 數值 | `abg_500Hz`, `abg_1000Hz`, `abg_2000Hz`, `abg_4000Hz` |
| ABG 缺失標記 | `abg_500Hz_missing`, `abg_1000Hz_missing`, `abg_2000Hz_missing`, `abg_4000Hz_missing` |
| ABG 截尾 標記 | `abg_500Hz_censored`, `abg_1000Hz_censored`, `abg_2000Hz_censored`, `abg_4000Hz_censored` |

Task2 不使用：

- `ac_mean`
- `bc_mean`
- `abg_mean`
- `PTA` 作為 hearing type rule

### Task3 特徵

Task3 共 16 個 特徵。

| 特徵 group | 欄位 |
|---|---|
| Raw 數值 | `tymp_Vea`, `tymp_peak_daPa`, `tymp_peak_mmho`, `tymp_Width_daPa` |
| real zero 標記 | `tymp_Vea_real_zero`, `tymp_peak_daPa_real_zero`, `tymp_peak_mmho_real_zero`, `tymp_Width_daPa_real_zero` |
| 缺失 zero 標記 | `tymp_Vea_missing_zero`, `tymp_peak_daPa_missing_zero`, `tymp_peak_mmho_missing_zero`, `tymp_Width_daPa_missing_zero` |
| NP zero 標記 | `tymp_Vea_np_zero`, `tymp_peak_daPa_np_zero`, `tymp_peak_mmho_np_zero`, `tymp_Width_daPa_np_zero` |

## 3. 訓練輸出檔案

每個 seed / experiment 會輸出到：

```text
results_v74/five_runs/<run_config>/<exp>_seed_<seed>/
```

| 檔案 | 意義 |
|---|---|
| `best_model.pth` | 最佳 checkpoint。 |
| `training_history.csv` | 每個 epoch 的 train loss、val loss、mean macro-F1、task 訓練步驟。 |
| `eval_summary.json` | configured 驗證 結果。若 meta support 啟用，可能使用 驗證 support。 |
| `eval_summary_no_support.json` | 部署 no-support 結果，固定 `support=None`。 |
| `eval_summary_locked_test_no_support.json` | 若使用 locked split manifest，輸出 locked test no-support 結果。 |
| `per_class_metrics.csv` | configured 驗證 的 per-class precision/recall/F1/support。 |
| `per_class_metrics_no_support.csv` | no-support 驗證 的 per-class precision/recall/F1/support。 |
| `per_class_metrics_locked_test_no_support.csv` | locked test no-support per-class metrics。 |
| `prediction_rows.csv` | configured 驗證 的逐筆預測。 |
| `prediction_rows_no_support.csv` | no-support 驗證 的逐筆預測。 |
| `prediction_rows_locked_test_no_support.csv` | locked test no-support 逐筆預測。 |

## 4. 驗證 摘要 差異

| 檔案 | 使用情境 |
|---|---|
| `eval_summary.json` | configured 驗證，用來觀察訓練設定本身的驗證表現。 |
| `eval_summary_no_support.json` | 部署 驗證，固定 `support=None`，更接近 dashboard 實際推論。 |
| `eval_summary_locked_test_no_support.json` | locked test split 上的 no-support final 評估。 |

常見欄位：

| 欄位 | 意義 |
|---|---|
| `loss` | 驗證 loss |
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
| `expert_target` | 臨床規則 / expert consistency target |
| `expert_confidence` | reward 信心度；越低代表 rule 證據 越不確定 |

`prediction_rows_no_support.csv` 固定 `support=None`，是 dashboard/部署 最應優先對照的逐筆結果。

## 6. Paper export 輸出

正式位置：

```text
paper_v74/tables/
```

目前已產生：

| 檔案 | 意義 |
|---|---|
| `summary_all_configs.csv` | configured 驗證 跨 run config 彙整。 |
| `all_runs_all_configs.csv` | configured 驗證 所有 run 詳細資料。 |
| `per_class_all_configs.csv` | configured 驗證 per-class 彙整。 |
| `summary_no_support_all_configs.csv` | no-support 跨 run config 彙整。 |
| `all_runs_no_support_all_configs.csv` | no-support 所有 run 詳細資料。 |
| `per_class_no_support_all_configs.csv` | no-support per-class 彙整。 |
| `history_all_configs.csv` | 所有訓練 history 彙整。 |
| `run_configs.csv` | run config metadata。 |

## 7. 缺失-aware rule 基準 輸出

正式位置：

```text
results_v74/rule_baselines_phase3/
```

| 檔案 | 說明 |
|---|---|
| `rule_baseline_summary.csv` | 各 task 的 forced、covered-only、暫不判讀-as-error 指標。 |
| `rule_baseline_summary.json` | 同上，JSON 格式。 |
| `rule_baseline_predictions.csv` | 每筆 ear-level 基準 預測與 證據狀態。 |
| `rule_baseline_evidence_summary.csv` | 各 證據狀態 的數量與準確率。 |

重要欄位：

| 欄位 | 意義 |
|---|---|
| `forced_pred_label` | 即使缺值也由 rule 硬判出的 label。 |
| `abstain_pred_label` | 若 基準 證據 不足，顯示 `INSUFFICIENT_EVIDENCE`。 |
| `baseline_covered` | 是否屬於 基準 有足夠證據可正式判斷。 |
| `evidence_status` | `complete_evidence`、`bc_missing`、`no_bc_data`、`nr_or_censored`、`missing_tymp_data` 等。 |
| `forced_correct` | forced 預測 是否等於人工 label。 |
| `abstain_correct` | 暫不判讀-aware 預測 是否等於人工 label。 |

## 8. 傳統機器學習基準 輸出

正式位置：

```text
results_v74/ml_baselines/
```

| 檔案 | 意義 |
|---|---|
| `ml_baseline_summary.csv` | 各 task / model / seed 的整體 metrics。 |
| `ml_baseline_per_class.csv` | per-class metrics。 |
| `ml_baseline_confusion_matrices.json` | confusion matrix。 |
| `ml_baseline_略過_models.csv` | optional boosting 缺套件時的 略過 log。 |
| `ml_baseline_manifest.json` | 執行資訊與輸出摘要。 |

## 9. Error 分析 輸出

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
| `rule_model_conflict_summary.csv` | 規則-模型衝突 彙整。 |
| `task2_clinical_subgroups.csv` | Task2 臨床 subgroup 指標。 |
| `task2_confusion_focus.csv` | Task2 confusion pair 與 臨床 標記。 |
| `error_analysis_manifest.json` | 分析摘要。 |

## 10. Split 實驗協定 輸出

正式位置：

```text
results_v74/split_protocol/
```

| 檔案 | 意義 |
|---|---|
| `split_manifest.csv` | 每筆資料的 train/val/test assignment。 |
| `split_summary.csv` | 每個 task / seed / split 的分布摘要。 |
| `split_protocol_manifest.json` | split protocol 執行資訊。 |

## 11. 人工缺失 輸出

正式位置：

```text
results_v74/artificial_missingness/
```

| 檔案 | 意義 |
|---|---|
| `artificial_missingness_summary.csv` | 不同人工缺失 scenario 的整體指標。 |
| `artificial_missingness_predictions.csv` | 各 scenario 的逐筆預測。 |
| `artificial_missingness_manifest.json` | 執行資訊與 checkpoint。 |

注意：`scenario=none` 是全資料 no-support inference，不等同 正式 驗證 split。

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

## 13. 混合式決策 評估 輸出

正式位置：

```text
paper_v74/hybrid_evaluation/
```

| 檔案 | 意義 |
|---|---|
| `hybrid_summary.csv` | model_only、rule_forced、rule_暫不判讀_as_error、混合式決策_rule_first 比較。 |
| `hybrid_predictions.csv` | 逐筆 混合式決策 預測。 |
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

- 規則 label
- 暫不判讀 規則 label
- 證據狀態
- 規則信心度
- 規則-模型衝突
- 警示原因

## 15. `all/` 同步狀態

`all/` 目前已同步主要程式、md、CSV 與正式輸出。同步 manifest：

```text
all/sync_manifest_v74.json
```

注意：根目錄的 `all.zip` 是舊壓縮檔，不代表目前最新版 `all/`。

## 16. 閱讀結果建議順序

1. 先看 `paper_v74/tables/summary_no_support_all_configs.csv`：部署 情境整體表現。
2. 再看 `paper_v74/tables/per_class_no_support_all_configs.csv`：少數類別是否有學到。
3. 再看 `results_v74/rule_baselines_phase3/rule_baseline_summary.csv`：缺值感知 rule 基準。
4. 再看 `results_v74/ml_baselines/ml_baseline_summary.csv`：classical ML 對照。
5. 再看 `paper_v74/error_analysis/subgroup_metrics.csv` 與 `rule_model_conflict_summary.csv`：錯誤來源。
6. 最後看 `results_v74/artificial_missingness/artificial_missingness_summary.csv` 與 `results_v74/model_profile/model_profile.csv`：缺失壓力測試與部署成本。

---

## 2026-06-18 最新輸出欄位更新

新增正式輸出與欄位如下：`paper_v74/tables/main_hybrid_summary.csv` 彙整 規則/模型/混合式決策的 Task1/Task2/Task3 macro-F1 與 coverage；`rule_true_model_conflicts.csv` / `rule_true_model_conflict_summary.csv` 記錄 `rule_label`、`true_label`、`model_pred`、`model_correct`、`rule_correct`、`rule_model_agree`、`rule_true_model_conflict_type`；per-class tables 新增 `low_support_flag`、`low_support_reason`、`low_support_threshold`；`calibration_summary.csv`、`calibration_ece_bins.csv`、`confidence_threshold_curve.csv` 分別提供 ECE/Brier/門檻 coverage；`deployment_profile.csv` 記錄 neural、混合式決策 規則優先、classical ML latency rows。

大型中間檔如 `prediction_rows_all.csv` 與 `hybrid_predictions.csv` 屬可重建分析輸出；正式打包時可用 `sync_all_outputs_v74.py --skip-large-predictions` 跳過。

## 2026-06-21 強健性 / Importance / 封包 輸出欄位摘要

此歷史段落已整理為可讀摘要：

- `artificial_missingness_summary.csv`：記錄 checkpoint、task、scenario、strategy，以及 Accuracy、Macro-F1、Balanced Accuracy、Macro Sensitivity、Macro Specificity。
- `artificial_missingness_degradation_summary.csv`：以 complete-資料 scenario 為 基準，計算各缺失情境的性能下降。
- `artificial_missingness_per_class.csv`：記錄每類 TP/FP/FN/TN、Sensitivity、Specificity、Precision、F1 與 support。
- `evidence_compensation_summary.csv`：整理缺失證據、替代證據來源與性能下降關係。
- `feature_importance_baseline.csv`：特徵重要性 / ablation 的 基準 performance。
- `feature_group_ablation_summary.csv`：特徵群組 mean-mask inference-time ablation 摘要。
- `feature_group_permutation_detail.csv`：每次 permutation repeat 的詳細結果，屬大型 逐列 分析輸出。
- `feature_group_permutation_importance.csv`：特徵群組 permutation importance 摘要。
- `hybrid_summary.csv` / `main_hybrid_summary.csv`：混合式決策 strategies 與 信心度-gated 混合式決策 指標，包含 `low_confidence_abstain_rate`。
- `sync_manifest_v74.json`：記錄 封包 inventory，例如 `include_checkpoints`、`checkpoint_files_in_all`、`large_prediction_files_in_all`。

重要欄位：

- `scenario_family`：缺失情境家族，例如 `bc_missing`、`tymp_missing`、`np_like`。
- `removed_evidence`：該 scenario 移除的臨床證據。
- `fallback_evidence`：該 scenario 下模型或 混合式決策 可依賴的替代證據。
- `macro_sensitivity` / `macro_specificity`：各類 sensitivity/specificity 的 macro average。
- `*_drop_from_complete`：相對 complete-資料 基準 的性能下降。
- `low_confidence_abstain_rate`：信心度-gated 混合式決策 因低 模型信心度 而輸出 `INSUFFICIENT_EVIDENCE` 的比例。

<!-- 2026-06-27-p0-p4-final-alignment -->

## 2026-06-27 P0-P4 收斂後最新說明

### 最新研究定位
本專案目前應定位為：**Missing-aware clinical-rule-guided hybrid decision-support framework for ear-level hearing classification under incomplete audiological evidence**。核心不是單純證明 Transformer 或 MetaIRL 分數較高，而是證明在不完整聽力學資料下，系統能結合臨床規則、模型預測、規則/模型衝突警示、低信心度暫不判讀與 locked-test 可追溯性。

### 目前新增或強化的證據鏈
- `ml_baseline_locked_test_summary_5seed.csv`：補齊 classical ML 在 locked-test 上的公平比較。
- `main_hybrid_summary*.csv`：新增 rule availability、rule failure、rule correction、model fallback success、rule-model conflict、warning rate 等臨床價值指標。
- `hybrid_threshold_sweep*.csv`：用 threshold sweep 說明 confidence gate 不是任意指定。
- `hybrid_strategy_mcnemar*.csv`：提供 strategy-level paired comparison / McNemar 檢定。
- `clinical_error_taxonomy_summary.csv`：將 true label / rule label / model prediction 整理成臨床錯誤 taxonomy。
- `calibration_policy_summary.csv`：把 confidence threshold curve 收斂成可引用的 policy summary。
- `statistical_summary_all_configs.csv`：新增 bootstrap CI 欄位。
- `artificial_missingness_v74.py` 與 `feature_importance_v74.py`：主流程預設改用 `run_13_tiny_m15_bc_dominant_r010/no_meta_seed_*` 作為 primary 5-seed 分析基礎。

### 論文敘事應避免過度主張
- 不應主張 device shift 已完成驗證，除非後續補上 device label 或 cross-device validation。
- 不應主張 Transformer / MetaIRL / IRL 本身是唯一主要貢獻；這些應寫成 neural component 或 ablation factor。
- 不應主張 hybrid 一定全面擊敗 clinical rule；應主張 hybrid 的價值在 rule coverage、model fallback、conflict warning 與 incomplete-data tolerance。
- 邊緣端 latency/profile 目前可作補充，不作本輪主軸；若要寫 IEEE IoT/IoMT，需以概念性 IoMT decision-support pipeline、missingness robustness 與 clinical warning 為主。

### 建議論文主線
1. Subject-level / locked-test traceable split 建立可信評估基礎。
2. Clinical rule baseline 建立可解釋診斷參考。
3. Neural model 補足 rule 無法完整判斷或資料不完整的案例。
4. Hybrid rule-guided framework 整合 rule decision、model fallback、warning 與 abstention。
5. Missingness robustness、feature importance、calibration 與 statistical tests 支撐系統可靠性。

<!-- /2026-06-27-p0-p4-final-alignment -->


<!-- 2026-06-27-new-output-labels -->

## 2026-06-27 新增輸出檔案與欄位意義

| 檔案 | 意義 |
|---|---|
| `results_v74/ml_baselines/ml_baseline_locked_test_summary_5seed.csv` | Classical ML baseline 在 locked-test split 上的 5-seed 主表，用於和 neural / rule / hybrid 做公平比較。 |
| `results_v74/ml_baselines/ml_baseline_locked_test_per_class_5seed.csv` | Locked-test ML baseline 的 per-class precision / recall / F1 / support。 |
| `paper_v74/hybrid_evaluation/hybrid_threshold_sweep.csv` | No-support 情境下 rule confidence threshold 與 model confidence threshold 的 grid sweep。 |
| `paper_v74/hybrid_evaluation/hybrid_threshold_sweep_locked_test.csv` | Locked-test 情境下的 hybrid threshold sweep。 |
| `paper_v74/hybrid_evaluation/hybrid_strategy_mcnemar.csv` | No-support 情境下 model/rule/hybrid strategy 的 paired accuracy delta 與 McNemar p-value。 |
| `paper_v74/hybrid_evaluation/hybrid_strategy_mcnemar_locked_test.csv` | Locked-test 情境下的 strategy-level McNemar comparison。 |
| `paper_v74/error_analysis/clinical_error_taxonomy_summary.csv` | 將 rule unavailable、rule correct/model wrong、model correct/rule wrong、both wrong 等錯誤來源整理成臨床 taxonomy。 |
| `results_v74/calibration_analysis/calibration_policy_summary.csv` | 從 confidence threshold curve 中，在 coverage 下限下選出建議 threshold。 |
| `results_v74/calibration_analysis_locked_test/calibration_policy_summary.csv` | Locked-test calibration policy summary。 |

新增 hybrid 主表欄位：`rule_available_rate`、`rule_correct_when_available_rate`、`rule_failure_rate`、`rule_correction_rate`、`model_fallback_success_rate`、`rule_model_conflict_rate`、`warning_rate`。這些欄位用來回答「為什麼需要 rule-guided hybrid framework」，而不是只比較 macro-F1。

<!-- /2026-06-27-new-output-labels -->

<!-- 2026-07-01-task1-missingness-alignment -->
## 2026-07-01 Task1 Artificial Missingness Scenarios

`results_v74/artificial_missingness/artificial_missingness_summary.csv`、`artificial_missingness_degradation_summary.csv`、`artificial_missingness_per_class.csv` 與 `artificial_missingness_manifest.json` 現在包含 Task1 的人工缺失情境。

| scenario | 對應訓練策略 | removed_evidence | fallback_evidence | 意義 |
|---|---|---|---|---|
| `none` | baseline | none | not_applicable | 完整 Task1 PTA baseline。 |
| `task1_no_pta` | `mask_pta` | AC PTA summary | frequency-specific AC thresholds | 測試沒有 PTA summary 時，模型能否依靠各頻率 AC。 |
| `task1_no_high_freq` | `mask_high_freq` | AC 2000/4000 Hz and PTA summary | low-frequency AC thresholds | 測試高頻 AC 與 PTA 缺失時的 Task1 穩健性。 |
| `task1_no_low_freq` | `mask_low_freq` | AC 500/1000 Hz and PTA summary | high-frequency AC thresholds | 測試低頻 AC 與 PTA 缺失時的 Task1 穩健性。 |
| `task1_no_all_ac` | `mask_all_ac` | all Task1 AC evidence | missingness-imputed feature baseline only | 極端壓力測試，代表 Task1 AC evidence unavailable。 |

Task1 的人工缺失是將對應 AC/PTA 欄位設為缺失值後再進入既有 preprocessing；它用來對齊訓練端 feature masking。這和 Task2/Task3 的顯式 missing/NR/NP 旗標不同，解讀時要分開描述。
<!-- /2026-07-01-task1-missingness-alignment -->
## 2026-07-02 最新規則更新：Rule Evidence Score 與 Hybrid Gating

- Task2 clear ABG 現在定義為 `AC-BC >= 15 dB`。
- Task2 borderline ABG 現在定義為 `10 <= AC-BC < 15 dB`；borderline 不直接觸發 CHL/MHL，只扣 rule evidence score 0.1 並加上 `abg_borderline` warning。
- Task2 的 `rule_forced` 仍保留硬判 label；但正式 rule-first / hybrid 是否採用 rule，改由 `rule_confidence` / `rule_evidence_score` 門檻控制。
- Task2 score 起始為 1.0：缺 core AC 扣 0.15；6000/8000 兩者都缺扣 0.05；BC 部分缺失整組扣 0.3；no BC data 直接壓到 0.5；NR/censored 只加 warning，不當 missing。
- Task3 evidence score 納入 `tymp_Vea` 缺失檢查；Vea 缺失只扣 0.05 並加 warning，不直接改 A/B/C label。
- Task3 `peak_daPa` 缺失時 score 最高 0.5；`peak NP + compliance NP` 視為有效 B 型證據；`-300 < peak_daPa <= -150` 為 C 區間但扣 0.2，並保留 C/B compatible label。
- Hybrid rule-first 現在使用 score gating：`rule_confidence >= 0.8` 且 rule label 存在時採用 rule；score 不足時 fallback model；model confidence 低於門檻時可輸出 `INSUFFICIENT_EVIDENCE`。
- 新增或保留的輸出欄位包含 `rule_evidence_score`、`score_deductions`、`rule_confidence`、`warning_reasons`、`hybrid_decision_reason`。
- 注意：依目前資料暫存檢查，`AC-BC>=15` 搭配「任一頻率有 clear ABG 即判 CHL/MHL」會明顯拉低 Task2 rule-forced 表現；這是規則定義的結果，不是流程錯誤，後續若要改善需再討論是否加入「至少兩個非 censored ABG 頻率」等條件。
## 2026-07-02 新增輸出檔案與欄位意義

### Rule / Hybrid 類
- rule_contribution_summary*.csv：rule_available_rate 代表 rule 可產生有效 label 的比例；rule_coverage_rate 代表 rule evidence score 達到 rule-first 門檻的比例；rule_correction_rate 代表 rule 修正 model 錯誤的比例；model_fallback_success_rate 代表 fallback 給 model 後 model 正確的比例；warning_rate 代表該群組中有 clinical warning 或低信心 warning 的比例。
- hybrid_explainability_summary*.csv：decision_reason 說明 hybrid 採用 rule 或 model 的原因，例如 rule_score_ge_threshold、rule_score_ge_threshold_with_warning、model_fallback_low_rule_score、model_fallback_no_rule_prediction、abstain_low_model_confidence。
- main_method_comparison*.csv：將各 strategy 的 macro-F1、coverage、abstain、fallback、conflict、warning 做 mean/std/min/max 彙整，適合放入論文主比較表。

### Missingness 類
- missingness_degradation_summary.csv：以 complete data 為基準，計算不同人工缺失情境的 accuracy、macro-F1、sensitivity、specificity 下降量。
- missingness_evidence_compensation_summary.csv：整理缺失情境、被移除證據、替代證據來源與各策略平均表現。
- missingness_hybrid_reason_summary.csv：統計每種缺失情境下 hybrid decision reason 的比例與正確率。

### Feature importance 類
- feature_importance_summary.csv：整合 mean-mask ablation 與 permutation importance 的 feature group 重要性。
- feature_group_importance_summary.csv：跨 importance source 彙整 feature group 的平均重要性與 within-task rank。
- feature_missingness_link_summary.csv：將 feature group importance 與對應 artificial missingness scenario 的性能下降做連結。

### Calibration 類
- calibration_summary_paper*.csv：整理 accuracy、mean_confidence、confidence_accuracy_gap、calibration_direction、ECE、Brier score、selected threshold、coverage 與 abstain rate。
