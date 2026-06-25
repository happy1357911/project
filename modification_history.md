# Modification History

<!-- 2026-06-25-p0-p4-gpt-analysis-update -->

## 2026-06-25 P0-P4 GPT Analysis Response Implementation

**Modification Time**: 2026-06-25 19:50

**Reason and Meaning**: Implemented the P0-P4 response to the latest GPT all.zip analysis and professor feedback. The changes add package reproducibility semantics, locked-test calibration, missingness-safe hybrid behavior, decision safety analysis, dashboard warning visibility, and repeated-seed statistical summaries.

**Files and Scope**:
- `sync_all_outputs_v74.py`: package type, raw/checkpoint/large-row manifest flags, optional raw data sync, obsolete rule-baseline requirement removal.
- `run_all_v74.py`: package flags, locked-test calibration, error analysis `mode all`, manifest verification, statistical summary verification.
- `run_locked_test_v74.py`: locked-test error analysis, three-way conflict, calibration, model confidence threshold.
- `calibration_analysis_v74.py`: prediction-row schema validation, `model_confidence` fallback, clear missing-input errors.
- `artificial_missingness_v74.py`: scenario-level model fallback, Task2 ABG borderline set to 10 dB, fallback scenarios recorded in manifest.
- `hybrid_evaluation_v74.py`: `hybrid_decision_reason`, `hybrid_warning_reasons`, decision reason summary outputs.
- `error_analysis_v74.py`: `decision_safety_summary.csv`, robust confusion grouping.
- `dashboard_three_tasks_metaIRL_v74.py`: model confidence, hybrid decision reason, low-confidence abstain.
- `evaluate_v74.py`: `statistical_summary_all_configs.csv` with mean/std/SEM/95 percent CI.

**Concrete Changes**: Added analysis-only/execution-ready package semantics; added locked-test post-processing; made hybrid safer under incomplete artificial-missingness scenarios; added safety and CI tables; updated dashboard and final output verification.

**Before vs After**: Previously, all.zip could be mistaken as re-runnable, locked-test lacked calibration/error-analysis post-processing, hybrid decisions lacked explicit reasons, and main result tables lacked CI. Now package semantics, tables, hybrid/error/dashboard outputs, and statistical summaries are explicit and traceable.

**Improved Functionality**: Reproducibility manifest, locked-test ECE/Brier/threshold curves, hybrid decision reason summary, decision safety summary, statistical CI summary, dashboard low-confidence abstention.

**Follow-up**: Medium priority: after running the formal full pipeline, inspect locked-test calibration, decision safety, and statistical summary values before writing the final paper narrative.

<!-- /2026-06-25-p0-p4-gpt-analysis-update -->


## 2026-06-25 External Run Config JSON

**時間**: 2026-06-25 16:44:30

**修改目的與意義**: 將訓練參數組合從 `train_v74.py` 的硬編碼常數中獨立出來，改由 `configs/run_configs_v74.json` 管理，讓 15 組 recommended 與 5 組 ablation 後續可直接改 JSON，不必修改主訓練程式。

**修改檔案與位置**:
- `configs/run_configs_v74.json`: 新增外部參數組合檔。
- `train_v74.py`: 新增 JSON loader、schema validation、`--run_config_file` / `--run-config-file` CLI。
- `run_all_v74.py`: 新增 `--run-config-file`，並傳給主訓練與 locked-test runner。
- `run_locked_test_v74.py`: 新增 `--run-config-file`，並傳給 `train_v74.py`。
- `sync_all_outputs_v74.py`: 將 `configs/run_configs_v74.json` 加入 `all/` 同步清單。

**修改內容**:
- JSON 內含 `model_size_configs`、`missing_aug_profiles`、`run_configs`、`ablation_run_configs`。
- `train_v74.py` 會在啟動時讀取 JSON，展開成原本訓練使用的完整 run config dict。
- validation 會檢查 recommended=15、ablation=5、run name 不重複、`missing_aug_p > 0`、`d_model % nhead == 0`、model size / missing profile 名稱存在。
- `run_all_v74.py --dry-run` 與 `run_locked_test_v74.py --dry-run` 會顯示 config file 已傳入下游訓練命令。

**修改前後差異**:
- 修改前：要調整參數組合必須直接改 `train_v74.py` 的 `RUN_CONFIGS` / `ABLATION_RUN_CONFIGS`。
- 修改後：預設由 `configs/run_configs_v74.json` 管理參數組合；Python 程式負責讀取、驗證與展開。

**驗證**:
- `python -m json.tool configs/run_configs_v74.json` 通過。
- project2 環境中 `load_run_config_sets()` 顯示 recommended=15、ablation=5、all=20。
- `run_02_base_m10_balanced_r015` 仍存在於 config list，預設 checkpoint resolver 不受影響。
- `python run_all_v74.py --compile-only --run-config-file configs/run_configs_v74.json` 通過。
- `python run_all_v74.py --dry-run --skip-model-profile --run-locked-test --locked-allow-overwrite --run-config-file configs/run_configs_v74.json` 顯示主訓練與 locked-test 都帶入 config file。
- sync smoke 顯示 `configs/run_configs_v74.json` 已被複製到 formal package 目標。

**後續優化**: Low: 若未來要快速比較不同 config 檔，可再新增 `configs/README.md` 或多個 preset JSON，例如 `run_configs_fast.json`、`run_configs_submission.json`。

## 2026-06-25 Task2/Task3 Rule and Source Alignment

**時間**: 2026-06-25 16:24:15

**修改目的與意義**: 依使用者確認後的規則修正，將 Task2/Task3 對齊修正版 6/24 資料來源，並修正 NR、ABG、Task3 tympanogram rule 與 hybrid rule-first gating，使訓練、baseline、dashboard、missingness、feature importance、profile 的行為一致。

**修改檔案與位置**:
- `clinical_rules_v74.py`: NR parser、Task2 hearing type rule、Task2 evidence/completeness、Task3 tympanogram rule。
- `preprocessing_v74.py`: Task2/Task3 source、Task2 feature list、evidence status、clinical warning summary。
- `train_v74.py`: Task2/Task3 source、Task2 feature list、local NR helper、training missingness augmentation。
- `ml_baselines_v74.py`: Task2/Task3 source、Task2 feature list。
- `baseline_rules_v74.py`: Task2/Task3 source。
- `hybrid_evaluation_v74.py`: `complete_for_rule` rule-first gating。
- `artificial_missingness_v74.py`: `complete_for_rule` hybrid gating、Task2 high-frequency AC NR scenario。
- `dashboard_three_tasks_metaIRL_v74.py`: dashboard hybrid decision 使用 `complete_for_rule`。
- `model_profile_v74.py`: hybrid decision latency profile 使用 `complete_for_rule`。
- `error_analysis_v74.py`: 保留並解析 `complete_for_rule` 欄位。
- `feature_importance_v74.py`: Task2 六頻 AC 與 high-frequency AC feature groups。

**修改內容**:
- Task2/Task3 source 改為 `task2_3_pure_data(6_24).xlsx`。
- `xxNR` 才視為 numeric NR；替換為對應頻率設備上限並設 NR flag，純 `NR` 不視為有效 numeric NR。
- Task2 AC model features 擴充為 500/1000/2000/4000/6000/8000 Hz 與六頻 AC NR flags；ABG 維持 500/1000/2000/4000 Hz。
- Task2 rule 改用 `abs(AC-BC)>10`；AC abnormal 檢查六頻；6000/8000 任一存在即可滿足 high-frequency completeness；8-10 dB ABG 只作 borderline warning。
- Task3 rule 改為 peak-based A/C/B：`>-150` 為 A、`-300~ -150` 為 C 且 confidence 0.6、`<=-300` 為 B；NP peak/compliance 為 B。
- Hybrid rule-first、dashboard、artificial missingness、model profile 均改成只有完整 rule evidence 且達 confidence threshold 才採 rule。

**修改前後差異**:
- 修改前：Task2/Task3 預設讀舊 CSV；Task2 model 只吃四頻 AC；ABG 用單向差值；NR 與 missing/censored 在不同模組中的完整性語意不一致；Task3 使用舊 peak threshold；hybrid 只看 `baseline_covered`。
- 修改後：預設讀 6/24 修正版 xlsx；Task2 六頻 AC 進 model；ABG 使用絕對差；NR 被視為 measured-but-censored 且完整；Task3 rule 與教授定義一致；hybrid/data dashboard/profile 的 rule-first 條件一致。

**驗證**:
- `python -m py_compile` 通過所有 root Python 檔。
- `python run_all_v74.py --compile-only` 通過。
- `python run_all_v74.py --dry-run --skip-model-profile --run-locked-test --locked-allow-overwrite` 顯示完整流程順序正確。
- `baseline_rules_v74.py` smoke：Task2 n=382、coverage=0.3874、forced_macro_f1=0.8130；Task3 n=270、coverage=1.0000、forced_macro_f1=0.9531。
- `split_protocol_v74.py` smoke：assignment_rows=10390、summary_rows=9。

**後續優化**: Medium: 完整重跑後需重新檢查 Task2 NR/censored 導致的 MHL/SNHL conflict，並用新 `complete_for_rule` 欄位重看 hybrid rule rate 與 error analysis。


## 2026-06-21 P0-P2 Robustness / Explainability / Packaging

????
- `checkpoint_utils_v74.py`????? checkpoint resolver?
- `artificial_missingness_v74.py`??? degradation summary?per-class sensitivity/specificity?evidence compensation?confidence-gated hybrid strategy?
- `feature_importance_v74.py`??? feature group permutation importance ? inference-time ablation?
- `hybrid_evaluation_v74.py`??? `hybrid_rule_first_confidence_gate`?`low_confidence_abstain_rate`?
- `train_v74.py`??? missingness augmentation strategy weights???? checkpoint metadata?
- `model_profile_v74.py`??? checkpoint resolver??? summary merge key ???
- `dashboard_three_tasks_metaIRL_v74.py`?checkpoint selection ???? resolver?
- `run_all_v74.py`??? feature importance step?confidence threshold ???clean sync????????
- `sync_all_outputs_v74.py`???????????? checkpoint??? clean ? package inventory?
- Markdown/README????????????? edge profile ???????

????
- ???????? `py_compile`?
- `hybrid_evaluation_v74.py` ?? sample predictions smoke run???? strategy ??? summary/main summary?
- `artificial_missingness_v74.py` ?? Task3 single checkpoint smoke run??? summary/predictions/degradation ??????
- `train_v74.py` ????? weighted strategy sampler ? augmentation tensor ???
- `model_profile_v74.py` ?? one-checkpoint + hybrid prediction smoke run??????? config_name dtype merge error?
- `run_all_v74.py --dry-run` ????????

## 2026-06-21 Final Verification Addendum

????
- Root 19 ? `.py` ??????? `py_compile`?
- `python run_all_v74.py --compile-only` ????file coverage ?? `checkpoint_utils_v74.py` ? `feature_importance_v74.py`?
- `python run_all_v74.py --dry-run` ??????????????? missingness?feature importance?model profile?clean sync?verify?
- `sync_all_outputs_v74.py --root . --all-dir .codex_tmp/phase4_sync_smoke --clean --skip-large-predictions` ??? smoke??? md ?????checkpoint ??????
- `feature_importance_v74.py` ???? checkpoint smoke run??? baseline?ablation?permutation importance?manifest ?????

???????????????????????? root ??? `paper_v74/`?`results_v74/` ????? `run_all_v74.py` ?????

## 2026-06-21 RUN_CONFIGS 15 + ABLATION 5 Masking Update

????
- `train_v74.py`?`RUN_CONFIGS` ?? 15 ??`ABLATION_RUN_CONFIGS` ?? 5 ??
- `train_v74.py`??? `MODEL_SIZE_CONFIGS`?`MISSING_AUG_PROFILES`?`missing_aug_profile()`?`make_run_config()`?
- `train_v74.py`?training missingness augmentation ?? Task1??? `mask_pta`?`mask_high_freq`?`mask_low_freq`?`mask_all_ac`?
- `checkpoint_utils_v74.py` / `run_all_v74.py`??? checkpoint ?? `run_02_base_m10_balanced_r015/full_seed_0/best_model.pth`?
- `sync_all_outputs_v74.py`???????? 15 ? run ??? 5 ? ablation ???
- README / Markdown / ????????

????
- `python -m py_compile train_v74.py checkpoint_utils_v74.py run_all_v74.py sync_all_outputs_v74.py` ???
- `RUN_CONFIGS=15`?`ABLATION_RUN_CONFIGS=5`??? 20?
- ?? config ?????
- ?? config ? `missing_aug_p > 0`?
- ?? config ? `missing_aug_strategy_weights` ?? Task1/Task2/Task3?
- ?? config ? `d_model % nhead == 0`?
- Task1 augmentation smoke?tensor shape ??? masking ??????
- `run_all_v74.py --dry-run --skip-train --skip-model-profile --skip-sync-all` ????? checkpoint?


## 2026-06-21 Task2/Task3 Shared Data Source and NA Missing Tokens

**時間**: 2026-06-21

**修改目的與意義**: 將 Task2/Task3 原始資料來源改為 `task2_3_pure_data.csv`，並明確把 `NA`、`na`、`N/A`、`n/a` 視為缺失值，避免字串型 NA 被誤當成有效 label 或有效臨床特徵。因新檔副檔名為 `.csv` 但實際內容為 Excel/OpenXML，也補上不依賴 `openpyxl` 的共用讀檔邏輯。

**修改檔案與位置**:
- `clinical_rules_v74.py`: `MISSING_TOKENS`
- `preprocessing_v74.py`: `TASK_INFO`、`load_tabular_data()`、xlsx parser、`clean_label_columns()`
- `train_v74.py`: Task2/Task3 `TASK_INFO`、資料讀取、label cleaning
- `ml_baselines_v74.py`: Task2/Task3 `TASK_INFO`、資料讀取、label cleaning
- `baseline_rules_v74.py`: Task2/Task3 CSV 常數與 `load_csv()`
- `artificial_missingness_v74.py`: `load_task_data()`

**修改內容**:
- Task2 與 Task3 皆改讀 `task2_3_pure_data.csv`。
- 新增 `preprocessing_v74.load_tabular_data()`，可讀一般 CSV，也可讀目前這種副檔名為 `.csv` 但內容為 xlsx 的檔案。
- `clean_label_columns()` 改用 `MISSING_TOKENS` 的 case-insensitive lookup，讓 `NA`/`N/A` 類 token 被 drop，而不是進入 LabelEncoder。

**修改前後差異**:
- 修改前：Task2/Task3 分別讀舊 CSV；若直接改成 `task2_3_pure_data.csv` 會因檔案內容是 xlsx 而讀取失敗；部分字串型 NA 沒有明確列入缺失 token。
- 修改後：全流程原始資料讀取支援新共同檔，且 NA 類 token 會一致地被視為缺失值。

**新增或改善功能**:
- 新增不依賴外部套件的 xlsx-in-csv 讀取支援。
- 提升 Task2/Task3 label 與 feature 缺失處理一致性。

**後續建議**:
- Medium: 針對新資料做人工 clinical rule review，尤其 Task2 的 `abg_borderline/no_bc_data` 與 Task3 的 `missing_tymp_data`，不要只依賴程式 rule 輸出。
