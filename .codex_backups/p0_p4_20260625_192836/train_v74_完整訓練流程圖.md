# train_v74.py 完整訓練流程圖

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

## 1. 主訓練流程

```mermaid
flowchart TD
    A["Start train_v74.py"] --> B["讀取三個 CSV"]
    B --> C1["preprocessing_v74.prepare_task1_dataframe"]
    B --> C2["preprocessing_v74.prepare_task2_dataframe"]
    B --> C3["preprocessing_v74.prepare_task3_dataframe"]

    C1 --> D1["Task1 轉 ear-level"]
    C2 --> D2["Task2 轉 ear-level + NR/missing/censored flags"]
    C3 --> D3["Task3 轉 ear-level + zero/NP/missing flags"]

    D1 --> E["建立 union_features"]
    D2 --> E
    D3 --> E

    E --> F["pad_and_clean + label encoding"]
    F --> G["case_id group train/valid split"]
    G --> H{"locked_split_manifest?"}
    H -- "Yes" --> H1["保留 locked test rows"]
    H -- "No" --> I["z-score normalization"]
    H1 --> I
    I --> J["建立 DataLoader"]
    J --> K["建立 MetaIRL Transformer"]

    K --> L["依 task_sampling_mode 建立 task step schedule"]
    L --> M["每個 epoch 依 schedule 訓練 Task1/Task2/Task3"]

    M --> N{"enable_meta?"}
    N -- "Yes" --> O["build_episode: support/query split"]
    N -- "No" --> P["整個 batch 作為 query"]
    O --> Q["model(X_query, task, support=support)"]
    P --> Q

    Q --> R["class-weighted CE loss"]
    Q --> S["reward head output"]
    S --> T["expert_consistency_target_and_confidence"]
    T --> U["reward MSE * expert_confidence"]
    R --> V["task_loss = exp(-log_var)*CE + log_var + reward_weight*reward_loss"]
    U --> V

    V --> W["backprop + optimizer step"]
    W --> X["epoch validation"]
    X --> Y["mean macro-F1 選 best checkpoint"]
    Y --> Z1["final configured validation"]
    Y --> Z2["final deployment no-support validation"]
    Y --> Z3["optional locked-test no-support validation"]
    Z1 --> A1["eval_summary.json / per_class_metrics.csv / prediction_rows.csv"]
    Z2 --> A2["eval_summary_no_support.json / per_class_metrics_no_support.csv / prediction_rows_no_support.csv"]
    Z3 --> A3["eval_summary_locked_test_no_support.json / per_class_metrics_locked_test_no_support.csv / prediction_rows_locked_test_no_support.csv"]
    A1 --> B1["aggregate_results"]
    A2 --> B1
    A3 --> B1
    B1 --> C4["summary*.csv/json + all_runs_metrics*.csv"]
```

## 2. Feature 處理說明

### Task1

```mermaid
flowchart LR
    A["row-level CSV"] --> B["right ear row"]
    A --> C["left ear row"]
    B --> D["ac_500Hz, ac_1000Hz, ac_2000Hz, ac_4000Hz, ac_PTA"]
    C --> D
    D --> E["label: hearing_degree_WHO_PTAbased"]
```

Task1 feature 數：5。

### Task2

```mermaid
flowchart LR
    A["row-level CSV"] --> B["preprocessing_v74"]
    B --> C["convert NR / missing / censored evidence"]
    C --> D["right ear row"]
    C --> E["left ear row"]
    D --> F["AC / BC / ABG values"]
    E --> F
    F --> G["NR / missing / censored flags"]
    G --> H["filter WNL/SNHL/CHL/MHL"]
```

Task2 feature 數：32。

Task2 不使用 `ac_mean`、`bc_mean`、`abg_mean` 作為模型核心 feature，也不使用它們作為 hearing type rule。

### Task3

```mermaid
flowchart LR
    A["row-level CSV"] --> B["preprocessing_v74"]
    B --> C["real_zero / missing_zero / np_zero flags"]
    C --> D["right ear row"]
    C --> E["left ear row"]
    D --> F["tymp raw values + flags"]
    E --> F
    F --> G["label: tymp_type A/B/C"]
```

Task3 feature 數：16。

## 3. 分頭原理

```mermaid
flowchart TD
    A["union feature vector: 49 dims"] --> B["NumericFeatureTokenizer"]
    B --> C["Transformer shared encoder"]
    C --> D["Task embedding + FiLM adapter"]
    D --> E1["Task1 classification head"]
    D --> E2["Task2 classification head"]
    D --> E3["Task3 classification head"]
    D --> R1["Task1 reward head"]
    D --> R2["Task2 reward head"]
    D --> R3["Task3 reward head"]
```

分頭的意思：

- 三個 task 共用 encoder。
- 每個 task 有自己的 classification head。
- 每個 task 也有自己的 reward head。
- `log_vars` 讓不同 task 的 loss 有 uncertainty weighting。
- meta support 存在時，prototype 會輔助 logits。
- no-support 或 dashboard 情境則不使用 support。

## 4. 根據的指標

訓練期間選 best checkpoint 主要看：

- 各 task / label 的 macro-F1 平均
- validation loss 用於 scheduler / early stopping 輔助

每個 final evaluation 輸出：

- accuracy
- macro-F1
- balanced accuracy
- AUC
- per-class precision / recall / F1 / support

分析時建議優先看：

1. `paper_v74/tables/summary_no_support_all_configs.csv`
2. `paper_v74/tables/per_class_no_support_all_configs.csv`
3. `paper_v74/error_analysis/subgroup_metrics.csv`
4. `paper_v74/error_analysis/rule_model_conflict_summary.csv`

## 5. 模型怎麼使用、順序

```mermaid
sequenceDiagram
    participant CSV as CSV files
    participant Prep as preprocessing_v74
    participant Split as Group split
    participant Model as MetaIRL Transformer
    participant Eval as Evaluation
    participant Export as Export/Analysis

    CSV->>Prep: 讀取 Task1/Task2/Task3
    Prep->>Prep: 轉 ear-level + 補 flags + evidence status
    Prep->>Split: 依 case_id 切 train/valid 或 locked test
    Split->>Model: train loaders
    Model->>Model: CE + reward confidence loss
    Model->>Eval: configured validation
    Model->>Eval: no-support validation
    Model->>Eval: optional locked-test no-support
    Eval->>Export: summary / per-class / prediction rows
```

## 6. No-support 與 dashboard

```mermaid
flowchart LR
    A["dashboard inference"] --> B["model(x, task_name, support=None)"]
    C["eval_summary_no_support.json"] --> D["support=None"]
    B --> E["deployment scenario"]
    D --> E
```

因此 dashboard 結果應優先和 no-support outputs 比較。dashboard 也會透過 `preprocessing_v74.clinical_warning_summary()` 顯示 rule label、abstain rule label、evidence status、rule confidence、rule-model conflict 與 warning reasons。

## 7. 後處理與論文輔助流程

以下腳本不在 training loop 裡，而是訓練後或獨立執行：

```mermaid
flowchart TD
    A["results_v74"] --> B["evaluate_v74.py"]
    B --> C["paper_v74/tables"]

    D["CSV files"] --> E["baseline_rules_v74.py"]
    E --> F["results_v74/rule_baselines_phase3"]

    D --> G["ml_baselines_v74.py"]
    G --> H["results_v74/ml_baselines"]

    A --> I["error_analysis_v74.py"]
    I --> J["paper_v74/error_analysis"]

    D --> K["split_protocol_v74.py"]
    K --> L["results_v74/split_protocol"]

    M["checkpoint"] --> N["artificial_missingness_v74.py"]
    N --> O["results_v74/artificial_missingness"]

    M --> P["model_profile_v74.py"]
    P --> Q["results_v74/model_profile"]

    A --> R["hybrid_evaluation_v74.py"]
    R --> S["paper_v74/hybrid_evaluation"]

    C --> T["sync_all_outputs_v74.py"]
    F --> T
    H --> T
    J --> T
    L --> T
    O --> T
    Q --> T
    S --> T
    T --> U["all/"]
```

目前狀態：

| 流程 | 狀態 |
|---|---|
| `evaluate_v74.py` no-support export | 正式 `paper_v74/tables` 已產生。 |
| `baseline_rules_v74.py` phase3 | 正式 `results_v74/rule_baselines_phase3` 已產生。 |
| `ml_baselines_v74.py` | 正式 `results_v74/ml_baselines` 已產生。 |
| `error_analysis_v74.py` | 正式 `paper_v74/error_analysis` 已產生。 |
| `split_protocol_v74.py` | 正式 `results_v74/split_protocol` 已產生。 |
| `artificial_missingness_v74.py` | 正式 `results_v74/artificial_missingness` 已產生。 |
| `model_profile_v74.py` | 正式 `results_v74/model_profile` 已產生。 |
| `hybrid_evaluation_v74.py` | 正式 `paper_v74/hybrid_evaluation` 已產生。 |
| `sync_all_outputs_v74.py` | `all/` 已同步主要程式、md、CSV 與正式輸出。 |

## 8. Single-task 與 ablation

```mermaid
flowchart LR
    A["--config_preset recommended"] --> B["5 formal RUN_CONFIGS"]
    C["--config_preset ablation"] --> D["tiny/small ABLATION_RUN_CONFIGS"]
    E["--config_preset all"] --> F["formal + ablation configs"]
    G["--experiments single_task"] --> H{"--single_task_target"}
    H --> I["Task1"]
    H --> J["Task2"]
    H --> K["Task3"]
    H --> L["all: expand to three aliases"]
```

## 9. Locked-test evaluation

```mermaid
flowchart TD
    A["split_protocol_v74.py"] --> B["split_manifest.csv"]
    B --> C["train_v74.py --locked_split_manifest"]
    C --> D["train rows"]
    C --> E["validation rows"]
    C --> F["locked test rows"]
    D --> G["training"]
    E --> H["early stopping / validation"]
    F --> I["locked_test_no_support final evaluation"]
    I --> J["eval_summary_locked_test_no_support.json"]
    I --> K["prediction_rows_locked_test_no_support.csv"]
```

## 10. 驗證狀態

目前已確認：

- 主要 `.py` 檔案可通過 `py_compile`。
- Task2 rule 對 CSV：174 rows、mismatch=0、uncertain evidence=75。
- no-support training outputs 已可產生。
- paper tables、ML baseline、rule baseline phase3、error analysis、split protocol、artificial missingness、model profile、hybrid evaluation 已有正式輸出。
- `all/` 已同步主要程式、md、CSV 與正式輸出。

注意：根目錄 `all.zip` 是舊壓縮檔，不代表目前最新版 `all/`。

---

## 2026-06-18 完整流程更新

目前完整流程由 `run_all_v74.py` 統一調度：compile → split protocol → train → rule baseline → ML baselines → error analysis with rule merge → evaluate paper tables → hybrid rule-first summary → calibration analysis → artificial missingness → deployment profile → optional locked-test runner → sync all → verify。正式 locked-test 不直接混入 `results_v74`，需用 `run_all_v74.py --run-locked-test --locked-allow-overwrite` 或獨立 `run_locked_test_v74.py --allow-overwrite`。

一般快速檢查建議先跑 `python run_all_v74.py --dry-run` 與 `python run_all_v74.py --compile-only`；正式全流程會很重，尤其 train、ML baseline、artificial missingness、locked-test。

## 2026-06-21 run_all_v74.py ?????
????????

1. `python -m py_compile ...`????? root Python?
2. `split_protocol_v74.py`??? grouped locked-test split manifest?
3. `train_v74.py`??? full/no_meta/no_irl/single_task?
4. `baseline_rules_v74.py`?clinical rule baseline?
5. `ml_baselines_v74.py`?classical ML baseline?
6. `error_analysis_v74.py`?true/rule/model ?? conflict ? subgroup analysis?
7. `evaluate_v74.py`?paper tables/figures?
8. `hybrid_evaluation_v74.py --mode no_support`?
9. `hybrid_evaluation_v74.py --mode locked_test`?
10. `calibration_analysis_v74.py`?
11. `artificial_missingness_v74.py`?
12. `feature_importance_v74.py`?
13. `model_profile_v74.py`?
14. optional `run_locked_test_v74.py`?
15. `sync_all_outputs_v74.py --clean --skip-large-predictions`?
16. `verify_key_outputs()`?

?????

```powershell
python run_all_v74.py
```

???? edge/model profile?

```powershell
python run_all_v74.py --skip-model-profile
```

## 2026-06-21 ???????15 ??? + 5 ??
??????????

- `RUN_CONFIGS = 15`?3 ??????base/small/tiny??? 5 ? masking profile?
- `ABLATION_RUN_CONFIGS = 5`?5 ??????????? training masking?
- `--config_preset recommended` ?? 15 ?????
- `--config_preset ablation` ?? 5 ????
- `--config_preset all` ?? 20 ??

?????? `missing_aug_p > 0`?? `missing_aug_strategy_weights` ???? `Task1`?`Task2`?`Task3`??? Task1 ?? masking strategy?`mask_pta`?`mask_high_freq`?`mask_low_freq`?`mask_all_ac`?

??????
- base: `run_01` ? `run_05`
- small: `run_06` ? `run_10`
- tiny: `run_11` ? `run_15`
- masking profile: `m05_balanced`?`m10_balanced`?`m15_bc_dominant`?`m15_tymp_dominant`?`m20_heavy_balanced`

?????
- `ablation_base_m10_equal_steps_r015`
- `ablation_base_m10_rr_k4_r015`
- `ablation_base_m10_support2_r015`
- `ablation_base_m20_high_reward_r020`
- `ablation_base_m30_stress_r010`

?? checkpoint ??? `run_02_base_m10_balanced_r015/full_seed_0/best_model.pth`??????? clinical masking baseline?
