# Modification History

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
