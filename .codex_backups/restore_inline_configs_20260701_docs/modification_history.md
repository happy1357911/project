# Modification History

<!-- 2026-06-27-md-language-cleanup -->

## 2026-06-27 Markdown 繁中敘述清理

**時間**：2026-06-27

**為何要修改（修改的意義）**：使用者要求檢查所有主要 Markdown/README 是否仍有不必要的中英文混雜，並將一般敘述改成繁體中文；專有名詞、程式碼區塊、流程圖 block 與 `rule.md` 不需要翻譯。

**修改位置（檔案位置）**：
- `README_v74.txt`
- `modification_history.md`
- `train_v74_完整訓練流程圖.md`
- `專案架構與模型使用說明.md`
- `對話紀錄.md`
- `程式碼與結果改善建議報告.md`
- `與原始程式碼差異比對.md`
- `輸出檔案Label與欄位意義說明.md`

**修改內容**：
- 將目前敘事、程式碼對齊狀態、P0-P4 回應、規則/資料來源、外部 config、完整流程與邊緣端 profile 延後等段落整理為自然繁體中文。
- 還原先前機械翻譯誤改的檔名、路徑、CLI 參數與輸出檔名，例如 `baseline_rules_v74.py`、`ml_baselines_v74.py`、`results_v74`、`--data_dir`、`prediction_rows.csv`。
- 清理 `輸出s`、`使用d`、`Beca使用`、`re主要` 等不自然殘留字串。
- 保留 `rule.md` 不修改；流程圖、mermaid、程式碼區塊與必要專有名詞維持原本英文。

**驗證**：已用 `rg` 掃描主要 Markdown/README，確認誤翻檔名、CLI 參數與殘留錯字沒有再出現；並抽看 README、專案架構文件與對話紀錄前段確認內容可讀。

**後續優化**：低優先級。若未來要把歷史紀錄也全部改成純繁中，可另開一輪專門整理；本次重點是 GPT/教授會優先讀到的現況段落與不應誤導的檔名/指令。

<!-- /2026-06-27-md-language-cleanup -->

<!-- 2026-06-26-distributed-split-configs -->

## 2026-06-26 分散式訓練 Split Config

已新增兩份 split config，讓 20 組 JSON-defined config 可以分散到兩台電腦執行：

- `configs/splits/run_configs_v74_part_A.json`：10 組 config，採奇數索引的均衡切分。
- `configs/splits/run_configs_v74_part_B.json`：10 組 config，與 part A 互補。

`train_v74.py` 現在支援 `schema_version = v1_split`。split JSON 會指向 `../run_configs_v74.json`，並透過 `include_config_names` 選擇要執行的 config；若名稱不存在或重複會直接報錯，不會自動回退到其他 split 檔。

建議的分散式訓練格式：

```powershell
python train_v74.py --data_dir . --results_dir results_v74_part_A --seeds 0,1,2,3,4 --experiments full,no_meta,no_irl,single_task --single_task_target all --config_preset all --run_config_file configs/splits/run_configs_v74_part_A.json --locked_split_manifest results_v74/split_protocol/split_manifest.csv
```

電腦 A 使用 `part_A` 與 `results_v74_part_A`；電腦 B 使用 `part_B` 與 `results_v74_part_B`。兩台都完成後，將 `results_v74_part_*/five_runs/` 底下的 config 資料夾合併到主電腦的 `results_v74/five_runs/`，再由主電腦執行後處理。

<!-- /2026-06-26-distributed-split-configs -->



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


## 2026-06-25 External Run Config JSON

**時間**: 2026-06-25 16:44:30

**修改目的與意義**: 將訓練參數組合從 `train_v74.py` 的硬編碼常數中獨立出來，改由 `configs/run_configs_v74.json` 管理，讓 15 組 recommended 與 5 組 ablation 後續可直接改 JSON，不必修改主訓練程式。

**修改檔案與位置**:
- `configs/run_configs_v74.json`: 新增外部參數組合檔。
- `train_v74.py`: 新增 JSON loader、schema 驗證、`--run_config_file` / `--run-config-file` CLI。
- `run_all_v74.py`: 新增 `--run-config-file`，並傳給主訓練與 locked-test runner。
- `run_locked_test_v74.py`: 新增 `--run-config-file`，並傳給 `train_v74.py`。
- `sync_all_outputs_v74.py`: 將 `configs/run_configs_v74.json` 加入 `all/` 同步清單。

**修改內容**:
- JSON 內含 `model_size_configs`、`missing_aug_profiles`、`run_configs`、`ablation_run_configs`。
- `train_v74.py` 會在啟動時讀取 JSON，展開成原本訓練使用的完整 run config dict。
- 驗證 會檢查 recommended=15、ablation=5、run name 不重複、`missing_aug_p > 0`、`d_model % nhead == 0`、model size / 缺失 profile 名稱存在。
- `run_all_v74.py --dry-run` 與 `run_locked_test_v74.py --dry-run` 會顯示 config file 已傳入下游訓練命令。

**修改前後差異**:
- 修改前：要調整參數組合必須直接改 `train_v74.py` 的 `RUN_CONFIGS` / `ABLATION_RUN_CONFIGS`。
- 修改後：預設由 `configs/run_configs_v74.json` 管理參數組合；Python 程式負責讀取、驗證與展開。

**驗證**:
- `python -m json.tool configs/run_configs_v74.json` 通過。
- 專案2 環境中 `load_run_config_sets()` 顯示 recommended=15、ablation=5、all=20。
- `run_02_base_m10_balanced_r015` 仍存在於 config list，預設 checkpoint resolver 不受影響。
- `python run_all_v74.py --compile-only --run-config-file configs/run_configs_v74.json` 通過。
- `python run_all_v74.py --dry-run --skip-model-profile --run-locked-test --locked-allow-overwrite --run-config-file configs/run_configs_v74.json` 顯示主訓練與 locked-test 都帶入 config file。
- sync smoke 顯示 `configs/run_configs_v74.json` 已被複製到 正式 封包 目標。

**後續優化**: Low: 若未來要快速比較不同 config 檔，可再新增 `configs/README.md` 或多個 preset JSON，例如 `run_configs_fast.json`、`run_configs_submission.json`。

## 2026-06-25 Task2/Task3 Rule and Source Alignment

**時間**: 2026-06-25 16:24:15

**修改目的與意義**: 依使用者確認後的規則修正，將 Task2/Task3 對齊修正版 6/24 資料來源，並修正 NR、ABG、Task3 tympanogram 規則 與 混合式決策 規則優先 gating，使訓練、基準、dashboard、缺失狀態、特徵重要性、profile 的行為一致。

**修改檔案與位置**:
- `clinical_rules_v74.py`: NR parser、Task2 聽損類型規則、Task2 證據/completeness、Task3 tympanogram 規則。
- `preprocessing_v74.py`: Task2/Task3 source、Task2 特徵 list、證據狀態、臨床警示 摘要。
- `train_v74.py`: Task2/Task3 source、Task2 特徵 list、local NR helper、訓練 缺失狀態 augmentation。
- `ml_baselines_v74.py`: Task2/Task3 source、Task2 特徵 list。
- `baseline_rules_v74.py`: Task2/Task3 source。
- `hybrid_evaluation_v74.py`: `complete_for_rule` 規則優先 gating。
- `artificial_missingness_v74.py`: `complete_for_rule` 混合式決策 gating、Task2 high-frequency AC NR scenario。
- `dashboard_three_tasks_metaIRL_v74.py`: dashboard 混合式決策 使用 `complete_for_rule`。
- `model_profile_v74.py`: 混合式決策 latency profile 使用 `complete_for_rule`。
- `error_analysis_v74.py`: 保留並解析 `complete_for_rule` 欄位。
- `feature_importance_v74.py`: Task2 六頻 AC 與 high-frequency AC 特徵群組。

**修改內容**:
- Task2/Task3 source 改為 `task2_3_pure_data(6_24).xlsx`。
- `xxNR` 才視為 numeric NR；替換為對應頻率設備上限並設 NR flag，純 `NR` 不視為有效 numeric NR。
- Task2 AC model 特徵 擴充為 500/1000/2000/4000/6000/8000 Hz 與六頻 AC NR 標記；ABG 維持 500/1000/2000/4000 Hz。
- Task2 rule 改用 `abs(AC-BC)>10`；AC abnormal 檢查六頻；6000/8000 任一存在即可滿足 high-frequency completeness；8-10 dB ABG 只作 邊界警示。
- Task3 規則 改為 peak-based A/C/B：`>-150` 為 A、`-300~ -150` 為 C 且 信心度 0.6、`<=-300` 為 B；NP peak/compliance 為 B。
- 混合式決策 規則優先、dashboard、人工缺失、model profile 均改成只有完整 rule 證據 且達 信心度門檻 才採 rule。

**修改前後差異**:
- 修改前：Task2/Task3 預設讀舊 CSV；Task2 model 只吃四頻 AC；ABG 用單向差值；NR 與 缺失/截尾 在不同模組中的完整性語意不一致；Task3 使用舊 peak 門檻；混合式決策 只看 `baseline_covered`。
- 修改後：預設讀 6/24 修正版 xlsx；Task2 六頻 AC 進 model；ABG 使用絕對差；NR 被視為 measured-but-截尾 且完整；Task3 規則 與教授定義一致；混合式決策/資料 dashboard/profile 的 規則優先 條件一致。

**驗證**:
- `python -m py_compile` 通過所有 root Python 檔。
- `python run_all_v74.py --compile-only` 通過。
- `python run_all_v74.py --dry-run --skip-model-profile --run-locked-test --locked-allow-overwrite` 顯示完整流程順序正確。
- `baseline_rules_v74.py` smoke：Task2 n=382、coverage=0.3874、forced_macro_f1=0.8130；Task3 n=270、coverage=1.0000、forced_macro_f1=0.9531。
- `split_protocol_v74.py` smoke：assignment_rows=10390、摘要_rows=9。

**後續優化**: Medium: 完整重跑後需重新檢查 Task2 NR/截尾 導致的 MHL/SNHL 衝突，並用新 `complete_for_rule` 欄位重看 混合式決策 rule rate 與 error 分析。


## 2026-06-21 P0-P2 強健性 / Explainability / Packaging

**時間**：2026-06-21

**為何要修改（修改的意義）**：依 GPT/教授建議補強 強健性、特徵重要性、混合式決策 信心度 gate、checkpoint resolver、封包 inventory 與文件同步，讓研究不只回答模型準確率，也能回答缺失資料、特徵依賴與部署 封包 可追溯性。

**修改的位置（檔案位置）**：
- `checkpoint_utils_v74.py`：新增 checkpoint resolver。
- `artificial_missingness_v74.py`：新增 degradation 摘要、per-class sensitivity/specificity、證據 compensation、信心度-gated 混合式決策 strategy。
- `feature_importance_v74.py`：新增 特徵群組 permutation importance 與 inference-time ablation。
- `hybrid_evaluation_v74.py`：新增 `hybrid_rule_first_confidence_gate` 與 `low_confidence_abstain_rate`。
- `train_v74.py`：新增 缺失狀態 augmentation strategy weights 與 checkpoint metadata。
- `model_profile_v74.py`：加入 checkpoint resolver，並修正 摘要 merge key 型別問題。
- `dashboard_three_tasks_metaIRL_v74.py`：checkpoint selection 對齊 resolver。
- `run_all_v74.py`：加入 特徵重要性 step、信心度門檻 傳遞與 clean sync 流程。
- `sync_all_outputs_v74.py`：補 封包 inventory、checkpoint 排除與 clean sync 語意。
- README / Markdown：同步目前流程與 邊緣端 profile 延後決策。

**修改的內容**：新增 強健性、explainability、混合式決策 safety 與 封包 可追溯相關輸出，並讓 checkpoint 選擇與 流程 順序更一致。

**修改前後的邏輯比對**：修改前 流程 偏向模型訓練與基本評估；修改後可同時分析缺失情境下的性能下降、特徵群重要性、混合式決策 是否採 rule、低信心是否 暫不判讀，以及 all 封包 是否包含可重跑所需檔案。

**修改後新增了什麼功能或優化了什麼功能**：新增 強健性/explainability/封包 三個層面的正式輸出與驗證流程。

**你認為還須優化**：Medium：完整重跑後需根據新結果重新整理論文主表與敘事。

## 2026-06-21 Final 驗證 Addendum

**時間**：2026-06-21

**為何要修改（修改的意義）**：補記 P0-P2 修改後的整體驗證結果，確認 root Python 檔、run_all dry-run、sync smoke 與 特徵重要性 smoke 均能通過。

**修改的位置（檔案位置）**：
- `run_all_v74.py`
- `sync_all_outputs_v74.py`
- `feature_importance_v74.py`
- 相關 README / Markdown 文件

**修改的內容**：確認 root 19 個 `.py` 可通過 `py_compile`；`run_all_v74.py --compile-only` 可列出完整 file coverage；`run_all_v74.py --dry-run` 可顯示 split、train、基準、error 分析、evaluate、混合式決策、校準、缺失狀態、特徵重要性、model profile、sync、verify 等步驟；sync smoke 可排除 checkpoint 與大型 逐筆預測列；特徵重要性 smoke 可產生 基準、ablation、permutation importance 與 manifest。

**修改前後的邏輯比對**：修改前只知道局部腳本可執行；修改後確認整體流程順序、輸出檢查與 封包 sync 行為一致。

**修改後新增了什麼功能或優化了什麼功能**：提升 流程 可檢查性與後續重跑前的信心。

**你認為還須優化**：Low：完成正式 full run 後再比對 `paper_v74/` 與 `results_v74/` 是否全部由最新程式產生。

## 2026-06-21 RUN_CONFIGS 15 + ABLATION 5 masking 更新

**時間**：2026-06-21

**為何要修改（修改的意義）**：將正式訓練組合擴充為 15 組 recommended 與 5 組 ablation，且每組都包含不同程度的 訓練 缺失狀態 masking，使研究能比較 model size、缺失狀態 profile、reward weight 與 ablation 設定。

**修改的位置（檔案位置）**：
- `train_v74.py`：`RUN_CONFIGS`、`ABLATION_RUN_CONFIGS`、`MODEL_SIZE_CONFIGS`、`MISSING_AUG_PROFILES`、`missing_aug_profile()`、`make_run_config()`。
- `checkpoint_utils_v74.py` / `run_all_v74.py`：預設 checkpoint resolver 對齊 `run_02_base_m10_balanced_r015/full_seed_0/best_model.pth`。
- `sync_all_outputs_v74.py`：同步 15 組 run 與 5 組 ablation 的 輸出 directories。
- README / Markdown / 對話紀錄：更新參數組合說明。

**修改的內容**：新增 base/small/tiny 三種模型規模與多種 缺失狀態 profile；所有 config 均要求 `missing_aug_p > 0`，並檢查 `missing_aug_strategy_weights` 涵蓋 Task1/Task2/Task3、`d_model % nhead == 0`。

**修改前後的邏輯比對**：修改前 設定組合較少且分散；修改後有 20 組系統化配置，可用 `--config_preset recommended`、`--config_preset ablation` 或 `--config_preset all` 控制。

**修改後新增了什麼功能或優化了什麼功能**：支援更完整的 masked parameter grid 與 ablation 比較。

**你認為還須優化**：Low：未來若常分散式訓練，可將 config split 產生器自動化。
## 2026-06-21 Task2/Task3 Shared 資料 Source and NA 缺失 Tokens

**時間**: 2026-06-21

**修改目的與意義**: 歷史紀錄：當時曾將 Task2/Task3 原始資料來源改為 `task2_3_pure_data.csv`，此後已被目前正式來源 `task2_3_pure_data(6_24).xlsx` 取代；並明確把 `NA`、`na`、`N/A`、`n/a` 視為缺失值，避免字串型 NA 被誤當成有效 label 或有效臨床特徵。因新檔副檔名為 `.csv` 但實際內容為 Excel/OpenXML，也補上不依賴 `openpyxl` 的共用讀檔邏輯。

**修改檔案與位置**:
- `clinical_rules_v74.py`: `MISSING_TOKENS`
- `preprocessing_v74.py`: `TASK_INFO`、`load_tabular_data()`、xlsx parser、`clean_label_columns()`
- `train_v74.py`: Task2/Task3 `TASK_INFO`、資料讀取、label cleaning
- `ml_baselines_v74.py`: Task2/Task3 `TASK_INFO`、資料讀取、label cleaning
- `baseline_rules_v74.py`: Task2/Task3 CSV 常數與 `load_csv()`
- `artificial_missingness_v74.py`: `load_task_data()`

**修改內容**:
- 歷史紀錄：Task2 與 Task3 當時曾改讀 `task2_3_pure_data.csv`；目前正式程式碼已改為 `task2_3_pure_data(6_24).xlsx`。
- 新增 `preprocessing_v74.load_tabular_data()`，可讀一般 CSV，也可讀目前這種副檔名為 `.csv` 但內容為 xlsx 的檔案。
- `clean_label_columns()` 改用 `MISSING_TOKENS` 的 case-insensitive lookup，讓 `NA`/`N/A` 類 token 被 drop，而不是進入 LabelEncoder。

**修改前後差異**:
- 歷史脈絡：當時的修改前狀態是 Task2/Task3 分別讀舊 CSV；若直接改成 `task2_3_pure_data.csv` 會因檔案內容是 xlsx 而讀取失敗；部分字串型 NA 沒有明確列入缺失 token。此段僅保留歷史脈絡，目前正式來源以 6/24 xlsx 為準。
- 修改後：全流程原始資料讀取支援新共同檔，且 NA 類 token 會一致地被視為缺失值。

**新增或改善功能**:
- 新增不依賴外部套件的 xlsx-in-csv 讀取支援。
- 提升 Task2/Task3 label 與 特徵 缺失處理一致性。

**後續建議**:
- Medium: 針對新資料做人工 臨床規則 review，尤其 Task2 的 `abg_borderline/no_bc_data` 與 Task3 的 `missing_tymp_data`，不要只依賴程式 rule 輸出。

## 2026-06-27 Statistical 摘要 Wide-Table Compatibility Fix

**日期**：2026-06-27

**修改原因**：`run_all_v74.py --skip-train --skip-split-protocol --skip-model-profile --package-type analysis_only` 在最終 verify 階段缺少 `paper_v74/tables/statistical_summary_all_configs.csv`。檢查後確認 `results_v74/five_runs` 的 20 組 config 與 per-config metrics 檔都存在，問題來自 `evaluate_v74.py` 的 `build_statistical_summary()` 只支援長表格式，但目前實際輸出的 `all_runs_metrics*.csv` 是寬表格式。

**修改檔案**：
- `evaluate_v74.py`
- `modification_history.md`

**修改內容**：
- `build_statistical_summary()` 保留原本長表格式支援。
- 新增寬表欄位解析，支援 `Task1__hearing_degree_WHO_PTAbased_acc`、`Task2__hearing_type_macro_f1`、`Task3__tymp_type_balanced_acc` 這類欄位。
- 將寬表 metric 正規化為 `accuracy`、`macro_f1`、`balanced_accuracy`。
- 修正 regex 解析順序與 non-greedy label matching，避免 `balanced_acc` 被誤切成 `label_col=..._balanced` 與 `metric=accuracy`。

**驗證結果**：
- `python -m py_compile evaluate_v74.py` 通過。
- `python evaluate_v74.py --results_dir results_v74 --paper_dir paper_v74 --heatmap_mode none` 通過。
- 已產生 `paper_v74/tables/statistical_summary_all_configs.csv`。
- 輸出列數為 2160，欄位包含 `config_name / exp_name / task / label_col / evaluation_scope / metric / n / mean / std / sem / ci95_low / ci95_high / ci95_half_width`。
- `metric` 分布：`accuracy=720`、`macro_f1=720`、`balanced_accuracy=720`。

**影響範圍**：只影響 paper table 的統計摘要輸出，不需要重跑 訓練 或 locked-test。
## 2026-06-27 all/ Structure and five_runs Sync Rule Fix

**日期**：2026-06-27

**修改原因**：使用者整理後的 `all/` 結構以 `paper_v74/five_runs/` 與 compact `results_v74/five_runs/` 為準；舊版程式仍會把 paper per-config 輸出攤平到 `paper_v74/run_*` / `paper_v74/ablation_*`，且 `sync_all_outputs_v74.py` 仍預設同步對話紀錄延伸檔與舊的 `requirements6_clean.txt`。

**修改檔案**：
- `evaluate_v74.py`
- `sync_all_outputs_v74.py`
- `run_all_v74.py`
- `modification_history.md`
- `對話紀錄.md`

**修改內容**：
- `evaluate_v74.py`：multi-run paper 輸出固定到 `paper_v74/five_runs/<config>/`，預測-row 補畫 heatmap 也改寫入 `paper_v74/five_runs/<config>/figs/`。
- `evaluate_v74.py`：`paper_v74/manifest.json` 新增 `runs_output_dir` 與 `run_paper_dirs`，明確記錄正式 per-config paper 位置。
- `sync_all_outputs_v74.py`：`DEFAULT_DIRS` 改為同步 `paper_v74/five_runs`，移除舊的攤平 run/ablation 清單。
- `sync_all_outputs_v74.py`：新增 compact `results_v74/five_runs` 同步，只複製 config-level 摘要 / metrics / manifest，不複製 seed 子資料夾、checkpoint 或大型 逐筆預測列。
- `sync_all_outputs_v74.py`：預設同步檔案移除 `requirements6_clean.txt`、`cleanup_candidates_20260618.md`、`gpt_suggestions_action_plan_v74.md`；manifest 會列出這些 record-extension md 是刻意排除。
- `run_all_v74.py`：移除舊的 13b flattened paper run sync，verify 改檢查 `paper_v74/five_runs`、`all/paper_v74/five_runs`、`all/results_v74/five_runs/run_configs.csv` 與 `five_runs_sync_mode`。

**驗證結果**：
- `python -m py_compile evaluate_v74.py sync_all_outputs_v74.py run_all_v74.py` 通過。
- `python evaluate_v74.py --results_dir results_v74 --paper_dir paper_v74 --heatmap_mode none` 通過。
- `paper_v74` 根目錄攤平 run/ablation 資料夾數量為 0，`paper_v74/five_runs` 有 20 組。
- `paper_v74/manifest.json` 已包含 `runs_output_dir`，且 `run_dirs_count=20`、`run_paper_dirs_count=20`。
- `all_sync_test` 測試同步通過：`paper_v74/five_runs=20`、`results_v74/five_runs=20`、compact 檔案=221、checkpoint=0、大型 逐筆預測列=0。
- `all_sync_test` 不包含 `cleanup_candidates_20260618.md`、`gpt_suggestions_action_plan_v74.md`、`requirements6_clean.txt`。

**注意事項**：本次 `all_sync_test` 為測試資料夾，沒有覆蓋使用者整理好的正式 `all/`。正式覆蓋需由使用者確認後再執行 `sync_all_outputs_v74.py --all-dir all --clean ...` 或完整 `run_all_v74.py`。
## 2026-06-27 paper_v74 manifest sync addendum

**日期**：2026-06-27

**修改原因**：檢查 `all/` 同步結果時發現 root 的 `paper_v74/manifest.json` 存在，但 `all/paper_v74/manifest.json` 未同步。依最小改動原則，只補同步清單，不改其他流程。

**修改檔案**：
- `sync_all_outputs_v74.py`
- `modification_history.md`
- `對話紀錄.md`

**修改內容**：
- 將 `paper_v74/manifest.json` 加入 `DEFAULT_FILES`，使 `sync_all_outputs_v74.py` 會同步 paper 封包 manifest。

**驗證方式**：
- `python -m py_compile sync_all_outputs_v74.py`
- 重新執行 `sync_all_outputs_v74.py --root . --all-dir all --clean --package-type analysis_only --skip-large-predictions`

## 2026-06-27 sync manifest raw-資料 範圍 wording

**修改時間**：2026-06-27

**為何要修改（修改的意義）**：最新 GPT 分析指出 `all/` 可能包含 目前使用的原始資料，但 manifest 只用 `include_raw_data` 布林值描述，容易讓 分析 封包 是否含 原始資料 的語意不清楚。本次只補強同步 manifest 的 raw-資料 範圍 說明，不改同步檔案選擇邏輯。

**修改的檔案與位置**：
- `sync_all_outputs_v74.py`：封包類型 note 與 manifest 欄位。
- `run_all_v74.py`、`hybrid_evaluation_v74.py`、`artificial_missingness_v74.py`、`dashboard_three_tasks_metaIRL_v74.py`：僅調整 CLI/UI 顯示文字，使主軸改為 臨床規則引導 混合式決策 support。

**修改的內容**：新增 `raw_data_scope` manifest 欄位，並將 分析 封包 說明改成 目前使用的原始資料 可由 `--include-raw-data` 納入；同時微調少量使用者可見文字，避免把研究主軸寫成單純 MetaIRL/Transformer 或 邊緣端 部署。

**修改前後差異**：修改前只有 `include_raw_data`，較難表達 分析 封包 可部分包含 active 原始 輸入；修改後 manifest 會同時提供 `include_raw_data` 與 `raw_data_scope`，文字敘事也更清楚區分 分析 封包 與 可執行重跑 封包。

**改善的功能**：提升 `all/` 封包 給 GPT/教授檢查時的可追溯性，降低 原始資料 / checkpoint / 大型 逐筆預測列 是否包含的誤解。

**後續優化建議**：若未來要建立真正 可執行重跑 封包，應使用 `--package-type execution_ready` 並確認 checkpoint、大型 逐筆預測列 與 原始資料 都同步完成。

## 2026-06-27 P0 locked-test baseline / hybrid clinical value / statistics

**時間**：2026-06-27 21:55

**修改目的**：依照 GPT/教授回饋，補齊 locked-test classical ML baseline、hybrid clinical value、threshold policy、paired comparison、error taxonomy 與 bootstrap CI，讓結果不只呈現單一 F1，而能說明 rule-guided hybrid 在 locked-test 與不完整資料情境下的臨床價值。

**修改檔案**：
- `ml_baselines_v74.py`：新增 split manifest locked-test evaluation。
- `hybrid_evaluation_v74.py`：新增 clinical value metrics、confidence threshold sweep、strategy McNemar paired comparison。
- `error_analysis_v74.py`：新增 clinical error taxonomy summary。
- `evaluate_v74.py`：在 statistical summary 補 bootstrap 95% CI。
- `checkpoint_utils_v74.py`：更新 primary checkpoint 至目前主分析設定 `run_13_tiny_m15_bc_dominant_r010/no_meta_seed_0`。
- `run_all_v74.py`：把 locked-test ML baseline、hybrid threshold sweep、5-seed missingness/feature-importance checkpoint glob 納入全流程。

**主要輸出**：
- `results_v74/ml_baselines/ml_baseline_locked_test_summary_5seed.csv`
- `paper_v74/hybrid_evaluation/hybrid_threshold_sweep.csv`
- `paper_v74/hybrid_evaluation/hybrid_threshold_sweep_locked_test.csv`
- `paper_v74/hybrid_evaluation/hybrid_strategy_mcnemar.csv`
- `paper_v74/hybrid_evaluation/hybrid_strategy_mcnemar_locked_test.csv`
- `paper_v74/error_analysis/clinical_error_taxonomy_summary.csv`
- `paper_v74/statistics/statistical_summary_all_configs.csv` 內新增 bootstrap CI 欄位

**小檢查 / 大檢查**：
- `ml_baselines_v74.py --help` 可看到 `--eval-mode` 與 `--split-manifest`。
- locked-test ML smoke test 可產出 locked-test baseline summary。
- hybrid clinical metrics、threshold sweep、McNemar、clinical taxonomy、bootstrap helper 皆以 synthetic dataframe 檢查通過。
- 全部 root Python 檔案 `py_compile` 通過。
- `python run_all_v74.py --compile-only` 通過。

**注意事項**：完整 `paper_v74` 與 `all` 新輸出仍需重新執行 `run_all_v74.py` 才會正式產生與同步。

## 2026-06-27 P1 missingness augmentation / calibration policy / primary-analysis alignment

**時間**：2026-06-27 22:03

**修改目的**：強化 robustness、explainability 與 confidence policy。把 Task2 的 ABG borderline 情境納入訓練 augmentation，並讓 missingness / feature importance 的主分析使用 5-seed primary checkpoint，而不是單一 seed。

**修改檔案**：
- `train_v74.py`：新增 Task2 `abg_borderline` missingness augmentation。
- `configs/run_configs_v74.json`：所有 Task2 missingness strategy weights 補上 `abg_borderline`，且各 profile 權重總和維持 1.0。
- `calibration_analysis_v74.py`：新增 `calibration_policy_summary.csv`，用 coverage 條件挑選較合理的 confidence threshold。
- `run_all_v74.py`：把 calibration policy summary、primary analysis checkpoint glob 與新檢查項納入全流程。

**主要輸出**：
- `results_v74/calibration_analysis/calibration_policy_summary.csv`
- `results_v74/calibration_analysis_locked_test/calibration_policy_summary.csv`
- `results_v74/artificial_missingness/artificial_missingness_summary.csv` 由 primary 5 seeds checkpoint 產生
- `results_v74/feature_importance/feature_importance_summary.csv` 由 primary 5 seeds checkpoint 產生

**小檢查 / 大檢查**：
- `configs/run_configs_v74.json` 通過 `python -m json.tool`。
- `train_v74.py`、`calibration_analysis_v74.py`、`run_all_v74.py` 通過 `py_compile`。
- 在 `project2` 環境中用 Task1/seed0 做 artificial missingness 與 feature importance smoke test，皆可完成。
- calibration policy smoke test 可產出 `calibration_policy_summary.csv`。
- 全部 root Python 檔案 `py_compile` 通過。
- `python run_all_v74.py --compile-only` 通過。

**注意事項**：既有 checkpoint 不會自動包含新加入的 `abg_borderline` training augmentation；若要讓模型本身真的用新 augmentation 訓練，必須重新跑 training。

## 2026-06-27 P2-P4 dashboard / documentation / narrative alignment

**時間**：2026-06-27 22:09

**修改目的**：把 P0-P1 新增的 formal analysis output 放進 dashboard 與文件敘事，並清掉容易讓 GPT/教授誤判的舊定位，例如過度主打 MetaIRL/Transformer、未證明 device shift、edge latency 當主軸等。

**修改檔案**：
- `dashboard_three_tasks_metaIRL_v74.py`：新增 `Formal analysis outputs` 區塊，可讀取 hybrid clinical value、threshold sweep、McNemar、clinical taxonomy、calibration policy 等表格。
- `README_v74.txt`
- `專案架構與模型使用說明.md`
- `程式碼與結果改善建議報告.md`
- `train_v74_完整訓練流程圖.md`
- `與原始程式碼差異比對.md`
- `輸出檔案Label與欄位意義說明.md`

**文件敘事更新**：
- 最新研究定位統一為 missing-aware clinical-rule-guided hybrid decision-support framework。
- 明確說明目前主軸是 incomplete audiological evidence 下的可靠性、rule/model/hybrid 決策價值、missingness robustness、feature importance、calibration 與 locked-test validation。
- 明確避免過度宣稱 device shift、Transformer/MetaIRL 是唯一貢獻、hybrid 永遠優於 clinical rule、edge latency 是本輪主軸。

**小檢查 / 大檢查**：
- `dashboard_three_tasks_metaIRL_v74.py` 通過 `py_compile`。
- 全部 root Python 檔案 `py_compile` 通過。
- `python run_all_v74.py --compile-only` 通過。
- 檢查 Markdown marker，確認本次文件更新以 UTF-8 繁體中文寫入，沒有問號化。

**注意事項**：Dashboard 與文件已能讀取/描述新輸出，但實際表格仍需重新執行 `run_all_v74.py` 後才會更新到 `paper_v74` 與 `all`。

## 2026-06-27 本機重訓分組與論文建議 MD 修正

**時間**：2026-06-27

**修改目的**：修正先前理解偏差。使用者要的是一份專門保存 GPT/教授後續論文建議的 Markdown 參考檔，而不是把論文敘事分散寫入多個 README。同時使用者確認要重新訓練本機負責的部分，使新加入的 `abg_borderline` training augmentation 能真正反映到 checkpoint。

**修改內容**：
- `configs/splits/run_configs_v74_part_A.json` 改為本機重訓 10 組：`run_01` 到 `run_07`，加上三個 m10 ablation config。
- `configs/splits/run_configs_v74_part_B.json` 改為互補 10 組：`run_08` 到 `run_15`，加上剩下兩個 ablation config。
- `gpt_suggestions_action_plan_v74.md` 改寫為「論文寫作建議整理 v7.4」，用途是保存 GPT/教授對後續論文敘事、claim、evidence、limitation 與章節安排的建議。

**後續操作重點**：本機應使用 `configs\splits\run_configs_v74_part_A.json` 重新訓練，輸出到 `results_v74_part_A`，之後再併回 `results_v74\five_runs` 或由全流程後處理讀取。
## 2026-06-29 run_all analysis checkpoint auto-scan 修正

**時間**：2026-06-29

**修改目的**：修正 `run_all_v74.py` 後處理階段不應固定依賴 `run_13_tiny_m15_bc_dominant_r010/no_meta_seed_*` 的問題。使用者指出流程應依照目前 `results_v74/five_runs` 下實際存在的 checkpoint 自動執行，而不是預設固定某幾組，否則 part_A / part_B 分段訓練時會被不存在的 config 卡住。

**修改檔案**：
- `run_all_v74.py`

**修改內容**：
- 新增 `glob` 掃描。
- `--primary-analysis-config` 預設維持 `auto`。
- `--primary-analysis-exp` 預設改為 `auto`。
- 預設 checkpoint glob 改為掃描 `results_v74/five_runs/*/*_seed_*/best_model.pth`。
- 若使用者指定 `--checkpoint` 或 `--analysis-checkpoint-glob`，仍優先尊重手動指定。
- 若使用者指定特定 `--primary-analysis-exp` 但找不到，仍保留 fallback 到 `full_seed_*` 的保守邏輯。

**驗證**：
- `python -m py_compile run_all_v74.py` 通過。
- 預設 `auto/auto` 實測可從目前 part_A 結果掃到 300 個 checkpoint。
- `python run_all_v74.py --compile-only` 通過。

**注意事項**：預設掃描所有現有 checkpoint 會比只跑單一 config / no_meta 更久，但符合「目前資料夾下有的都納入」的全流程邏輯。若只想快速測試，可手動加 `--primary-analysis-exp no_meta` 或 `--analysis-checkpoint-glob` 限縮。
## 2026-06-29 run_all Step 9/10 checkpoint-glob 修正

**時間**：2026-06-29

**修改目的**：修正前次 `run_all_v74.py` auto-scan checkpoint 後，將 300 個 checkpoint 路徑全部展開到 `--checkpoints`，造成 Windows `[WinError 206] 檔名或副檔名太長` 的問題。

**修改檔案**：
- `run_all_v74.py`

**修改內容**：
- Step 9 `artificial_missingness_v74.py` 改為使用 `--checkpoint-glob` 傳入 glob pattern。
- Step 10 `feature_importance_v74.py` 同樣改為使用 `--checkpoint-glob`。
- 保留 `--checkpoint` 手動指定單一 checkpoint 的能力。
- 不再把所有 checkpoint 路徑串成逗號清單傳入命令列。

**驗證**：
- `python -m py_compile run_all_v74.py` 通過。
- `python run_all_v74.py --dry-run ...` 顯示 Step 9/10 已使用 `--checkpoint-glob results_v74\five_runs\*\*_seed_*\best_model.pth`。
- 實際解析 checkpoint 數量為 300。
- `python run_all_v74.py --compile-only` 通過。
## 2026-07-01 Task1 artificial missingness alignment 與延伸 MD 刪除

**時間**：2026-07-01

**修改目的**：使用者確認 Task1 訓練端已有 missingness augmentation，但人工缺失分析端缺少對應 Task1 scenarios，因此補齊訓練設定與 robustness analysis 的對應關係。同時刪除兩個不再作為正式文件的延伸 MD：`gpt_suggestions_action_plan_v74.md`、`程式碼與結果改善建議報告.md`。

**程式修改**：
- `artificial_missingness_v74.py`：新增 Task1 scenarios：`task1_no_pta`、`task1_no_high_freq`、`task1_no_low_freq`、`task1_no_all_ac`，並補上 metadata 與人工遮蔽邏輯。
- `sync_all_outputs_v74.py`：移除已刪 MD 的同步/排除清單引用。

**刪除檔案**：
- `gpt_suggestions_action_plan_v74.md`
- `程式碼與結果改善建議報告.md`

**文件更新**：
- `README_v74.txt`
- `輸出檔案Label與欄位意義說明.md`
- `專案架構與模型使用說明.md`
- `train_v74_完整訓練流程圖.md`
- `與原始程式碼差異比對.md`

**驗證**：
- `python -m py_compile artificial_missingness_v74.py sync_all_outputs_v74.py` 通過。
- `task_scenarios("Task1")` 回傳 `none`、`task1_no_pta`、`task1_no_high_freq`、`task1_no_low_freq`、`task1_no_all_ac`。
- 小型 dataframe 檢查確認各 Task1 scenario 遮蔽欄位符合設計。
- Task1-only smoke test 成功產出 5 個 Task1 scenarios，每個 scenario 有 5 個策略列。
- `python run_all_v74.py --compile-only` 通過。

**注意事項**：Task1 的缺失是 AC/PTA feature masking，沒有 Task2/Task3 那種 explicit missing/NR/NP flag；論文敘述需要分開說明。