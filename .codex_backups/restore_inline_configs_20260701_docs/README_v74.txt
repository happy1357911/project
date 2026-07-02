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

<!-- 2026-06-26-current-code-alignment -->

## 2026-06-26 目前程式碼對齊狀態

本段是目前版本的優先採信段落；下方較早日期段落保留作為歷史紀錄。
目前程式碼對齊重點：
- Task1 資料來源：`task1_all_three_common14_v1.csv`。
- Task2/Task3 資料來源：`task2_3_pure_data(6_24).xlsx`。
- 目前特徵數：Task1 = 5、Task2 = 36、Task3 = 16、三任務 union = 53。
- Task2 模型輸入包含 AC 500/1000/2000/4000/6000/8000 Hz、六頻 AC NR 標記、BC 500/1000/2000/4000 Hz、BC NR/缺失標記、ABG 500/1000/2000/4000 Hz 的數值/缺失/截尾標記。
- Task2 規則使用 AC 500/1000/2000/4000 Hz，加上 6000/8000 Hz 至少一個高頻存在作為完整性條件；ABG 以 `abs(AC-BC)>10 dB` 判定，8-10 dB 只作邊界警示。
- Task3 規則：`peak_daPa <= -300` 判 B，`-300 < peak_daPa <= -150` 判 C 且信心度較低，`peak_daPa > -150` 判 A；NP peak 加 NP compliance 判 B。
- 混合式決策的規則優先策略只有在 `complete_for_rule=True`、`baseline_covered=True` 且 `rule_confidence` 達門檻時採用規則；否則使用模型，若模型信心度低於門檻則可暫不判讀。
- `configs/run_configs_v74.json` 為完整 20 組設定：15 組 `run_configs` 加 5 組 `ablation_run_configs`。
- `configs/splits/run_configs_v74_part_A.json` 與 `configs/splits/run_configs_v74_part_B.json` 各選 10 組，供兩台電腦分工訓練；沒有 A/B 自動回退。
- `all/` 預設是給 GPT/教授分析的 `analysis_only` 封包；可用 `--include-raw-data` 納入目前使用的原始資料，但未包含 checkpoint 與大型逐列預測時，仍不是完整可重跑封包。
- 本輪邊緣端/model profile 仍可用 `--skip-model-profile` 延後，不作為目前主流程必跑項。

Windows 主電腦完整刷新建議指令：

```powershell
python run_all_v74.py --skip-model-profile --run-locked-test --locked-allow-overwrite --package-type analysis_only
```

Linux / Docker 第二台電腦路徑提醒：請使用 `/`，例如 `configs/splits/run_configs_v74_part_B.json`，不要使用 Windows 的 `\`。

<!-- /2026-06-26-current-code-alignment -->

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

### 新增主要輸出
- `paper_v74/tables/statistical_summary_all_configs.csv`
- `results_v74/calibration_analysis_locked_test/`
- `paper_v74/hybrid_evaluation/hybrid_decision_reason_summary*.csv`
- `paper_v74/error_analysis/decision_safety_summary.csv`


缺值感知、臨床規則引導的聽力輔助判讀系統 v7.4

<!-- 2026-06-25-rule-source-update -->

## 2026-06-25 規則與資料來源更新

Task2/Task3 流程的正式資料來源已統一為 `task2_3_pure_data(6_24).xlsx`。這是 6/24 修正版，已透過 shared preprocessing 路徑供 preprocessing、訓練、ML baseline、rule baseline、split protocol、人工缺失、特徵重要性、混合式決策評估輸入，以及 dashboard/model-row logic 使用。

規則更新：
- `xxNR` 視為已測量但被截尾的證據，會替換成該頻率的設備上限值，並在規則完整性判定中視為完整；純 `NR` token 不視為有效 numeric NR 數值。
- Task2 模型輸入現在包含 AC 500/1000/2000/4000/6000/8000 Hz，以及六個頻率的 AC NR 標記；ABG 仍維持 500/1000/2000/4000 Hz。
- Task2 規則以 `abs(AC-BC) > 10 dB` 判定 ABG，並用 500/1000/2000/4000/6000/8000 Hz 檢查 AC 是否異常；6000/8000 Hz 只要其中一個存在，就視為高頻 AC 完整。
- Task2 的 8-10 dB ABG 只作邊界警示，不會單獨觸發 CHL/MHL；除非其他頻率已有明確 ABG。
- Task3 規則以 peak 為主：`peak_daPa > -150 => A`，`-300 < peak_daPa <= -150 => C` 且信心度 0.6，`peak_daPa <= -300 => B`；NP peak/compliance 證據仍判為 Type B。
- 混合式決策的規則優先策略只有在 `complete_for_rule=True`、`baseline_covered=True` 且規則信心度達到設定門檻時使用規則，其他情況交由模型處理。

本次更新後的 smoke check：
- `python run_all_v74.py --compile-only` 通過所有根目錄 Python 檔案。
- `python run_all_v74.py --dry-run --skip-model-profile --run-locked-test --locked-allow-overwrite` 顯示完整流程順序。
- `baseline_rules_v74.py` 使用修正版資料來源時，產生 Task2 n=382 與 Task3 n=270。

<!-- /2026-06-25-rule-source-update -->

<!-- 2026-06-25-external-run-configs -->

## 2026-06-25 外部 run config 更新

訓練參數組合已獨立到 `configs/run_configs_v74.json`。之後若要調整 15 組 recommended 或 5 組 ablation，不需要直接修改 `train_v74.py` 的常數區塊。

JSON 內容包含：
- `model_size_configs`：base / small / tiny model dimensions。
- `missing_aug_profiles`：缺失 augmentation 機率，以及 Task1/Task2/Task3 strategy weights。
- `run_configs`：15 組建議使用的 masked configurations。
- `ablation_run_configs`：5 組 ablation configurations。

預設用法：
```powershell
python run_all_v74.py --run-config-file configs/run_configs_v74.json --skip-model-profile --run-locked-test --locked-allow-overwrite
```

`train_v74.py`、`run_all_v74.py` 與 `run_locked_test_v74.py` 都支援外部 config file。`sync_all_outputs_v74.py` 也會把 `configs/run_configs_v74.json` 同步到 `all/`。

<!-- /2026-06-25-external-run-configs -->

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

更新日期：2026-06-08

本專案目前的研究重點：
在聽力資料存在缺失值、NR、截尾 AC/BC/ABG 與 tympanogram NP 的情況下，建立可用於三個任務的單耳層級多任務模型，並同時保留 no-support 部署評估、臨床規則 基準、傳統機器學習基準、子群/錯誤分析 與 split protocol。

============================================================
1. 目前主要程式
============================================================

1) train_v74.py
   主要訓練程式。負責資料前處理、單耳層級轉換、train/valid/locked-test split、神經網路多任務元件 訓練、configured/no-support/locked-test 評估、摘要 aggregation。

2) mtl_meta_irl_transformer.py
   神經網路元件架構。包含 數值 tokenizer、Transformer encoder、task embedding、FiLM adapter、classification heads、reward heads 與可作 ablation 的 prototype/meta-learning。

3) clinical_rules_v74.py
   共用臨床規則。Task2 聽損類型規則、Task3 tympanogram 規則、數值解析、NR/缺失/截尾 判讀集中在這裡。

4) baseline_rules_v74.py
   Rule-based 臨床 基準。

5) dashboard_three_tasks_metaIRL_v74.py
   Streamlit dashboard。推論時固定使用 support=None，並顯示 規則/模型/混合式決策、證據狀態、警示 與 衝突。

6) evaluate_v74.py
   將 結果 匯出為 論文表格與圖。程式已支援 no-support tables，但正式 paper_v74/tables 需重新執行後才會完整更新。

7) ml_baselines_v74.py
   傳統機器學習基準。支援 Logistic Regression、Random Forest、HistGradientBoosting、MLP；XGBoost/LightGBM/CatBoost 若未安裝會記錄為 略過。

8) error_analysis_v74.py
   Subgroup/error 分析。可分析 缺失、NR、截尾、NP、專家信心度分層、錯誤案例與 混淆配對。

9) split_protocol_v74.py
   Split 實驗協定 / locked test manifest。可建立 grouped train/valid split 與 可選 locked-test 切分。

============================================================
2. 目前輸入資料
============================================================

- task1_all_three_common14_v1.csv
- task2_3_pure_data(6_24).xlsx (Task2/Task3 6/24 修正版共用來源)

目前三個任務都轉為 ear-level sample：
- Task1：左右耳拆開，以單耳 AC/PTA 判斷聽損程度。
- Task2：左右耳拆開，以單耳 AC/BC/ABG 與 缺失/NR/截尾 標記 判斷 WNL/SNHL/CHL/MHL。
- Task3：左右耳拆開，以單耳 tympanogram raw 數值 與 real_zero/缺失_zero/np_zero 標記 判斷 A/B/C。

目前 union_特徵 = 53：
- Task1 特徵 = 5
- Task2 特徵 = 36
- Task3 特徵 = 16

============================================================
3. 目前實驗模式
============================================================

train_v74.py 預設會跑：
- RUN_CONFIGS：15 組 建議設定；另有 5 組 ABLATION_RUN_CONFIGS，皆由 `configs/run_configs_v74.json` 管理
- seeds：0,1,2,3,4
- experiments：full,no_meta,no_irl,single_task

模式意義：
- full：CE + reward + meta-learning，Task1/Task2/Task3。
- no_meta：CE + reward，不使用 meta support，Task1/Task2/Task3。
- no_irl：CE + meta-learning，reward_weight=0，Task1/Task2/Task3。
- single_task：可用 `--single_task_target Task1|Task2|Task3|all` 控制；`all` 會展開為 Task1/Task2/Task3 三個 single-task 別名。

重要提醒：
目前 single-task 別名 已支援 Task1、Task2、Task3；設定組合由 JSON 管理，包含 15 組 recommended 與 5 組 ablation。

============================================================
4. 常用指令
============================================================

正式訓練：
python train_v74.py --data_dir . --results_dir results_v74

只跑單一模式：
python train_v74.py --exp full --seed 0

只跑 ablation preset：
python train_v74.py --config_preset ablation --experiments full,no_meta,no_irl,single_task --seeds 0,1,2,3,4

匯出 paper tables / figures：
python evaluate_v74.py --results_dir results_v74 --paper_dir paper_v74

Rule 基準：
python baseline_rules_v74.py --data-dir . --output-dir results_v74/rule_baselines_phase2

傳統機器學習基準：
python ml_baselines_v74.py --data-dir . --output-dir results_v74/ml_baselines --seeds 0,1,2,3,4

Error 分析：
python error_analysis_v74.py --results_dir results_v74 --output_dir paper_v74/error_analysis --mode both

Split 實驗協定 / locked test：
python split_protocol_v74.py --data_dir . --output_dir results_v74/split_protocol --seeds 0,1,2,3,4 --locked_test_ratio 0.2

Dashboard：
streamlit run dashboard_three_tasks_metaIRL_v74.py

============================================================
5. 目前完成度
============================================================

已完成：
- Task2/Task3 臨床規則 集中到 clinical_rules_v74.py。
- 部署 no-support 評估 輸出 eval_summary_no_support.json、per_class_metrics_no_support.csv、prediction_rows_no_support.csv。
- run-level no-support aggregation：summary_no_support.csv/json、all_runs_metrics_no_support.csv。
- reward 信心度：缺失/NR/截尾 或 partial rule 證據 會降低 reward loss 影響。
- tiny/small ablation preset。
- 傳統機器學習基準 腳本。
- 子群/錯誤分析 腳本。
- split protocol / locked test manifest 腳本。

部分完成：
- evaluate_v74.py 已支援 no-support paper tables，但正式 paper_v74/tables 目前未完整重新產生。
- ml_baselines_v74.py 已可執行並通過 smoke test，但正式 results_v74/ml_baselines 尚未產生。
- error_analysis_v74.py 已可執行並通過 smoke test，但正式 paper_v74/error_analysis 尚未產生。
- split_protocol_v74.py 已可執行並通過 smoke test，但正式 results_v74/split_protocol 尚未產生。
- single_task aliases 已支援 Task1/Task2/Task3；正式輸出需確認三個 alias 的結果是否都已由最新訓練產生。

============================================================
6. 主要輸出
============================================================

訓練輸出：
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/best_model.pth
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/eval_summary.json
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/eval_summary_no_support.json
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/per_class_metrics.csv
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/per_class_metrics_no_support.csv
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/prediction_rows.csv
- results_v74/five_runs/<run_config>/<exp>_seed_<seed>/prediction_rows_no_support.csv

彙整輸出：
- results_v74/five_runs/<run_config>/summary.csv
- results_v74/five_runs/<run_config>/summary_no_support.csv
- results_v74/five_runs/<run_config>/all_runs_metrics.csv
- results_v74/five_runs/<run_config>/all_runs_metrics_no_support.csv

Paper export：
- paper_v74/tables/summary_all_configs.csv
- paper_v74/tables/all_runs_all_configs.csv
- paper_v74/tables/per_class_all_configs.csv
- no-support paper tables 需重新執行 evaluate_v74.py 後產生。

============================================================
7. 寫論文時的注意事項
============================================================

2026-06-21 資料-source 更新:
- Task2 and Task3 now 使用 `task2_3_pure_data(6_24).xlsx` as the shared corrected raw source.
- The file extension is `.csv`, but the 目前 file content is Excel/OpenXML; `preprocessing_v74.load_tabular_data()` handles both normal CSV and this xlsx-in-csv case.
- 新增 Task2 standard ear-level rows = 384; rule/label mismatch = 27. 主要 mismatch contexts are `abg_borderline`, `no_bc_data`, `bc_missing`, `ac_missing`, and `nr_or_censored`.
- 新增 Task3 valid A/B/C ear-level rows = 271; many rows have `missing_tymp_data`, so rule coverage and covered-only performance must be interpreted separately.

============================================================
8. 2026-06-11 正式 結果 position
============================================================

目前 recommended paper claim:
Audiologist-guided, ear-level, task-adaptive hearing classification under 缺失 / NR / 截尾 audiological measurements.

Primary 結果 should 使用 部署 no-support 輸出 because the dashboard 使用 support=None:
- paper_v74/tables/summary_no_support_all_configs.csv
- paper_v74/tables/per_class_no_support_all_configs.csv
- paper_v74/error_analysis/

目前 解讀:
- Do not overclaim that Transformer, MetaIRL, or IRL clearly beats 臨床規則.
- no_meta is close to or better than full on averaged no-support 結果.
- HistGradientBoosting and RandomForest 基準 are close to the 神經網路模型.
- Task2 CHL and Task3 Type C are rare classes, so macro-F1 must be read with support and 子群/錯誤分析.

新增 工作流程:
- train_v74.py supports --single_task_target Task1|Task2|Task3|all.
- train_v74.py supports --locked_split_manifest and locked-test no-support 輸出.
- error_analysis_v74.py writes rule_model_conflicts.csv and rule_model_conflict_summary.csv.
- sync_all_outputs_v74.py syncs selected 正式 source 檔案 and 輸出 into all/.

============================================================
9. 2026-06-11 缺值感知 rule 基準
============================================================

臨床規則 基準 now reports both forced 預測 and 缺值感知 暫不判讀 預測.

Run:
python baseline_rules_v74.py --data-dir . --output-dir results_v74/rule_baselines_phase3

輸出:
- rule_baseline_summary.csv/json
- rule_baseline_predictions.csv
- rule_baseline_evidence_summary.csv

解讀:
- forced_pred_label: rule still gives a label for every row.
- 暫不判讀_pred_label: rows with insufficient 證據 become INSUFFICIENT_EVIDENCE.
- coverage: proportion of rows where the rule had complete/usable 證據.
- 暫不判讀_rate: proportion of rows where the rule should not be treated as a confident real 基準 決策.
- Task2 keeps BC 缺失 / no BC / NR / 截尾 rows in n_total; they are not removed.

因此論文不建議主張「模型全面比臨床規則更準」。較合理的寫法是：混合式決策 規則優先 在保留 rule 可解釋性的同時，對 規則暫不判讀、不完整證據、模型回退 與 警示 進行可追蹤整合。較合理的研究主軸是：
- 缺失 / NR / 截尾 audiological measurements 下的模型穩定性。
- ear-level multi-task learning 是否能在少樣本 Task2/Task3 中提供輔助。
- configured 驗證 與 部署 no-support 的差異。
- 規則信心度 / reward 信心度 如何避免規則過度主導 CE loss。

## 2026-06-21 P0-P2 強健性 與 explainability 更新

此歷史段落已整理為可讀摘要：當時 `run_all_v74.py` 已整合 compile、grouped locked-test split、訓練、臨床規則 基準、傳統機器學習基準、三方 error 分析、paper table export、no-support/locked-test 混合式決策 評估、校準、人工缺失 強健性、特徵-group importance/ablation、model/部署 profile、all/ 封包 sync 與 verify。

常用指令：

```powershell
python run_all_v74.py
```

若本輪先不做 邊緣端 latency/profile，可使用：

```powershell
python run_all_v74.py --skip-model-profile
```

重要輸出包含：
- `results_v74/artificial_missingness/artificial_missingness_degradation_summary.csv`：缺失情境相對 complete 資料 的性能下降。
- `results_v74/artificial_missingness/evidence_compensation_summary.csv`：缺失證據與替代證據來源摘要。
- `results_v74/feature_importance/feature_group_permutation_importance.csv`：特徵群組 permutation importance。
- `results_v74/feature_importance/feature_group_ablation_summary.csv`：inference-time 特徵群組 mean-mask ablation。
- 混合式決策 評估 中的 `hybrid_rule_first_confidence_gate`：rule 不完整或信心不足時回退 model，model 低信心時可輸出 `INSUFFICIENT_EVIDENCE`。
- `sync_all_outputs_v74.py --clean`：同步 all/ 封包，預設可排除 `.pth/.pt/.ckpt` 與大型逐列預測。

## 2026-06-21 RUN_CONFIGS 15 + ABLATION 5 masking 更新

此歷史段落已整理為可讀摘要：當時將訓練組合擴充為 15 組 recommended 與 5 組 ablation。recommended 覆蓋 base/small/tiny 三種模型大小與 5 種 masking profile；ablation 也都保留 訓練 masking，只是 masking 程度、reward weight 或 schedule 不同。

目前最新狀態：完整 config 已外部化到 `configs/run_configs_v74.json`，並新增 `configs/splits/run_configs_v74_part_A.json` 與 `configs/splits/run_configs_v74_part_B.json` 供兩台電腦各跑 10 組。預設偏好 checkpoint 仍是 `run_02_base_m10_balanced_r015/full_seed_0/best_model.pth`。

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

<!-- 2026-07-01-task1-missingness-alignment -->
## 2026-07-01 Task1 缺失分析對齊

本版已將人工缺失壓力測試的 Task1 情境補齊，使分析端與訓練端一致。Task1 訓練端原本已有 `mask_pta`、`mask_high_freq`、`mask_low_freq`、`mask_all_ac` 四種 missingness augmentation；現在 `artificial_missingness_v74.py` 也對應新增 `task1_no_pta`、`task1_no_high_freq`、`task1_no_low_freq`、`task1_no_all_ac`。

需要注意的是，Task1 的缺失處理是 AC/PTA 數值遮蔽，沒有像 Task2 的 BC/ABG missing flag 或 Task3 的 missing/NP flag 那樣完整的顯式缺失旗標。因此論文敘述應寫成 Task1 feature-masking robustness，而不是把三個 task 的缺失機制說成完全相同。

另外，`gpt_suggestions_action_plan_v74.md` 與 `程式碼與結果改善建議報告.md` 已刪除；目前正式說明以本 README、專案架構說明、輸出欄位說明、訓練流程圖、差異比對、`modification_history.md` 與 `對話紀錄.md` 為準。
<!-- /2026-07-01-task1-missingness-alignment -->
