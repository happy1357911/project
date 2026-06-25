<!-- 2026-06-26-distributed-split-configs -->

## 2026-06-26 Distributed Split Configs

Two split config files were added for running the 20 JSON-defined configs across two computers:

- `configs/splits/run_configs_v74_part_A.json`: 10 configs, odd-indexed balanced split.
- `configs/splits/run_configs_v74_part_B.json`: 10 configs, complementary split.

`train_v74.py` now supports `schema_version = v1_split`. A split JSON points to `../run_configs_v74.json` and selects configs by `include_config_names`. Missing names or duplicate names raise an error; there is no fallback to another split file.

Recommended distributed training pattern:

```powershell
python train_v74.py --data_dir . --results_dir results_v74_part_A --seeds 0,1,2,3,4 --experiments full,no_meta,no_irl,single_task --single_task_target all --config_preset all --run_config_file configs/splits/run_configs_v74_part_A.json --locked_split_manifest results_v74/split_protocol/split_manifest.csv
```

Use `part_B` and `results_v74_part_B` on the second computer. After both finish, merge the config directories under `results_v74_part_*/five_runs/` into one master `results_v74/five_runs/`, then run post-processing on the master computer.

<!-- /2026-06-26-distributed-split-configs -->
<!-- 2026-06-25-p0-p4-gpt-analysis-update -->

## 2026-06-25 P0-P4 GPT Analysis Response Update

This update was implemented after checking the current codebase against the latest GPT all.zip analysis and professor feedback. The goal is to move the project from a model-score-only framing toward a missing-aware, clinical-rule-guided, locked-test-traceable decision-support pipeline.

Completed updates:
- `sync_all_outputs_v74.py` now declares package type: `analysis_only`, `execution_ready`, or `custom`.
- The sync manifest explicitly records whether raw data, checkpoints, and large row-level prediction files are included.
- `run_all_v74.py` passes package flags to sync and verifies that `sync_manifest_v74.json` matches the run arguments.
- `run_locked_test_v74.py` now produces locked-test error analysis, three-way conflict tables, calibration summary, ECE bins, Brier score, and confidence-threshold curves.
- `run_all_v74.py` now creates `results_v74/calibration_analysis_locked_test/` and runs error analysis with `--mode all`.
- Hybrid outputs now include `hybrid_decision_reason`, `hybrid_warning_reasons`, and decision-reason summary CSVs.
- Artificial missingness now applies conservative scenario-level model fallback for incomplete or corrupted stress scenarios.
- Task2 artificial ABG borderline stress now uses 10 dB instead of 15 dB, so it is truly borderline rather than clear ABG.
- Error analysis now outputs `decision_safety_summary.csv`.
- Dashboard now displays model confidence, hybrid decision reason, and low-confidence abstention.
- `evaluate_v74.py` now outputs `statistical_summary_all_configs.csv` with mean, std, SEM, and clipped 95 percent CI.

Recommended formal refresh command:

```powershell
python run_all_v74.py --skip-model-profile --run-locked-test --locked-allow-overwrite --package-type analysis_only
```

Use this only when a re-runnable large package is required:

```powershell
python run_all_v74.py --skip-model-profile --run-locked-test --locked-allow-overwrite --package-type execution_ready
```

<!-- /2026-06-25-p0-p4-gpt-analysis-update -->

### Added Main Outputs
- `paper_v74/tables/statistical_summary_all_configs.csv`
- `results_v74/calibration_analysis_locked_test/`
- `paper_v74/hybrid_evaluation/hybrid_decision_reason_summary*.csv`
- `paper_v74/error_analysis/decision_safety_summary.csv`


MetaIRL Hearing AI v7.4

<!-- 2026-06-25-rule-source-update -->

## 2026-06-25 Rule and Data Source Update

Task2/Task3 pipeline source is now `task2_3_pure_data(6_24).xlsx`. This is the corrected 6/24 file and is used by preprocessing, training, ML baselines, rule baselines, split protocol, artificial missingness, feature importance, hybrid evaluation inputs, and dashboard/model-row logic through the shared preprocessing path.

Rule updates:
- `xxNR` is treated as measured-but-censored evidence, replaced by the frequency-specific equipment upper limit, and counted as complete for rule completeness. A pure `NR` token is not treated as a valid numeric NR value.
- Task2 model input now includes AC 500/1000/2000/4000/6000/8000 Hz plus AC NR flags for all six frequencies. ABG remains 500/1000/2000/4000 Hz.
- Task2 rule uses `abs(AC-BC) > 10 dB` for ABG, checks AC abnormality across 500/1000/2000/4000/6000/8000 Hz, and treats 6000/8000 as complete when at least one is present.
- Task2 8-10 dB ABG is a borderline warning only; it does not trigger CHL/MHL unless another frequency has clear ABG.
- Task3 rule is peak-based: `peak_daPa > -150 => A`, `-300 < peak_daPa <= -150 => C` with confidence 0.6, and `peak_daPa <= -300 => B`. NP peak/compliance evidence remains Type B.
- Hybrid rule-first now uses the rule only when `complete_for_rule=True`, `baseline_covered=True`, and rule confidence reaches the configured threshold; otherwise it uses the model.

Smoke checks after this update:
- `python run_all_v74.py --compile-only` passed for all root Python files.
- `python run_all_v74.py --dry-run --skip-model-profile --run-locked-test --locked-allow-overwrite` showed the full pipeline order.
- `baseline_rules_v74.py` on the corrected source produced Task2 n=382 and Task3 n=270.

<!-- /2026-06-25-rule-source-update -->

<!-- 2026-06-25-external-run-configs -->

## 2026-06-25 External Run Config Update

Training parameter combinations are now externalized into `configs/run_configs_v74.json`.

The JSON file contains:
- `model_size_configs`: base / small / tiny model dimensions.
- `missing_aug_profiles`: missingness augmentation probabilities and Task1/Task2/Task3 strategy weights.
- `run_configs`: the 15 recommended masked configurations.
- `ablation_run_configs`: the 5 ablation configurations.

Default usage:
```powershell
python run_all_v74.py --run-config-file configs/run_configs_v74.json --skip-model-profile --run-locked-test --locked-allow-overwrite
```

`train_v74.py`, `run_all_v74.py`, and `run_locked_test_v74.py` all support the external config file. `sync_all_outputs_v74.py` also copies `configs/run_configs_v74.json` into `all/`.

<!-- /2026-06-25-external-run-configs -->

<!-- 2026-06-20-edge-profile-deferred -->

## 2026-06-20 決策：暫不執行 edge 端估計

目前先不把 edge/deployment latency profile 作為本輪必跑項目。主流程優先保留模型訓練、locked-test、rule/model/hybrid、missingness robustness、calibration 與 baseline 結果；`model_profile_v74.py` 與 `results_v74/model_profile/` 先列為後續需要 edge / IoMT deployment 敘事時再單獨補跑的項目。

<!-- 2026-06-19-codex-update -->

## 2026-06-19 Full Pipeline Update

`run_all_v74.py` is now the primary complete pipeline runner. It builds the grouped locked-test split before training and passes `--locked_split_manifest` into `train_v74.py`, so the primary run can produce both no-support and locked-test no-support outputs.

Run with the project2 environment:
```powershell
& "C:\Users\ASUS\anaconda3\envs\project2\python.exe" run_all_v74.py --python "C:\Users\ASUS\anaconda3\envs\project2\python.exe"
```

Key expected outputs include `main_hybrid_summary.csv`, `main_hybrid_summary_locked_test.csv`, `three_way_conflict_summary.csv`, `ml_baseline_summary_5seed.csv`, `artificial_missingness_summary.csv`, and `model_profile_summary.csv`.

<!-- /2026-06-19-codex-update -->

更新日期：2026-06-08

本專案目前的研究重點：
在 hearing data 存在 missing value、NR、censored AC/BC/ABG 與 tympanogram NP 的情況下，建立可用於三個任務的 ear-level multi-task model，並同時保留 no-support deployment evaluation、clinical rule baseline、classical ML baseline、subgroup/error analysis 與 split protocol。

============================================================
1. 目前主要程式
============================================================

1) train_v74.py
   主要訓練程式。負責資料前處理、ear-level conversion、train/valid split、MetaIRL Transformer training、configured validation、deployment no-support evaluation、summary aggregation。

2) mtl_meta_irl_transformer.py
   模型架構。包含 numeric tokenizer、Transformer encoder、task embedding、FiLM adapter、classification heads、reward heads、prototype meta-learning。

3) clinical_rules_v74.py
   共用臨床規則。Task2 hearing type rule、Task3 tympanogram rule、數值解析、NR/missing/censored 判讀集中在這裡。

4) baseline_rules_v74.py
   Rule-based clinical baseline。

5) dashboard_three_tasks_metaIRL_v74.py
   Streamlit dashboard。推論時固定使用 support=None，對應 deployment no-support 情境。

6) evaluate_v74.py
   將 results 匯出為 paper tables 與 figures。程式已支援 no-support tables，但正式 paper_v74/tables 需重新執行後才會完整更新。

7) ml_baselines_v74.py
   Classical ML baseline。支援 Logistic Regression、Random Forest、HistGradientBoosting、MLP；XGBoost/LightGBM/CatBoost 若未安裝會記錄為 skipped。

8) error_analysis_v74.py
   Subgroup/error analysis。可分析 missing、NR、censored、NP、expert confidence bucket、錯誤案例與 confusion pairs。

9) split_protocol_v74.py
   Split protocol / locked test manifest。可建立 grouped train/valid split 與 optional locked test split。

============================================================
2. 目前輸入資料
============================================================

- task1_all_three_common14_v1.csv
- task2_3_pure_data(6_24).xlsx (Task2/Task3 corrected 6/24 shared source)

目前三個任務都轉為 ear-level sample：
- Task1：左右耳拆開，以單耳 AC/PTA 判斷聽損程度。
- Task2：左右耳拆開，以單耳 AC/BC/ABG 與 missing/NR/censored flags 判斷 WNL/SNHL/CHL/MHL。
- Task3：左右耳拆開，以單耳 tympanogram raw values 與 real_zero/missing_zero/np_zero flags 判斷 A/B/C。

目前 union_features = 49：
- Task1 features = 5
- Task2 features = 36
- Task3 features = 16

============================================================
3. 目前實驗模式
============================================================

train_v74.py 預設會跑：
- RUN_CONFIGS：5 組正式參數
- seeds：0,1,2,3,4
- experiments：full,no_meta,no_irl,single_task

模式意義：
- full：CE + reward + meta-learning，Task1/Task2/Task3。
- no_meta：CE + reward，不使用 meta support，Task1/Task2/Task3。
- no_irl：CE + meta-learning，reward_weight=0，Task1/Task2/Task3。
- single_task：目前只訓練 Task2，不是 Task1/Task2/Task3 三個獨立 single-task 都跑。

重要提醒：
第八階段已加入 tiny/small ablation config，但「三個任務各自 single-task」尚未完成。目前 single_task 仍是 Task2-only。

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

Rule baseline：
python baseline_rules_v74.py --data-dir . --output-dir results_v74/rule_baselines_phase2

Classical ML baseline：
python ml_baselines_v74.py --data-dir . --output-dir results_v74/ml_baselines --seeds 0,1,2,3,4

Error analysis：
python error_analysis_v74.py --results_dir results_v74 --output_dir paper_v74/error_analysis --mode both

Split protocol / locked test：
python split_protocol_v74.py --data_dir . --output_dir results_v74/split_protocol --seeds 0,1,2,3,4 --locked_test_ratio 0.2

Dashboard：
streamlit run dashboard_three_tasks_metaIRL_v74.py

============================================================
5. 目前完成度
============================================================

已完成：
- Task2/Task3 clinical rule 集中到 clinical_rules_v74.py。
- deployment no-support evaluation 輸出 eval_summary_no_support.json、per_class_metrics_no_support.csv、prediction_rows_no_support.csv。
- run-level no-support aggregation：summary_no_support.csv/json、all_runs_metrics_no_support.csv。
- reward confidence：missing/NR/censored 或 partial rule evidence 會降低 reward loss 影響。
- tiny/small ablation preset。
- classical ML baseline 腳本。
- subgroup/error analysis 腳本。
- split protocol / locked test manifest 腳本。

部分完成：
- evaluate_v74.py 已支援 no-support paper tables，但正式 paper_v74/tables 目前未完整重新產生。
- ml_baselines_v74.py 已可執行並通過 smoke test，但正式 results_v74/ml_baselines 尚未產生。
- error_analysis_v74.py 已可執行並通過 smoke test，但正式 paper_v74/error_analysis 尚未產生。
- split_protocol_v74.py 已可執行並通過 smoke test，但正式 results_v74/split_protocol 尚未產生。
- single_task 目前仍是 Task2-only，尚未完成 Task1/Task2/Task3 各自 single-task ablation。

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

2026-06-21 data-source update:
- Task2 and Task3 now use `task2_3_pure_data(6_24).xlsx` as the shared corrected raw source.
- The file extension is `.csv`, but the current file content is Excel/OpenXML; `preprocessing_v74.load_tabular_data()` handles both normal CSV and this xlsx-in-csv case.
- New Task2 standard ear-level rows = 384; rule/label mismatch = 27. Main mismatch contexts are `abg_borderline`, `no_bc_data`, `bc_missing`, `ac_missing`, and `nr_or_censored`.
- New Task3 valid A/B/C ear-level rows = 271; many rows have `missing_tymp_data`, so rule coverage and covered-only performance must be interpreted separately.

============================================================
8. 2026-06-11 formal result position
============================================================

Current recommended paper claim:
Audiologist-guided, ear-level, task-adaptive hearing classification under missing / NR / censored audiological measurements.

Primary result should use deployment no-support outputs because the dashboard uses support=None:
- paper_v74/tables/summary_no_support_all_configs.csv
- paper_v74/tables/per_class_no_support_all_configs.csv
- paper_v74/error_analysis/

Current interpretation:
- Do not overclaim that Transformer, MetaIRL, or IRL clearly beats clinical rules.
- no_meta is close to or better than full on averaged no-support results.
- HistGradientBoosting and RandomForest baselines are close to the neural model.
- Task2 CHL and Task3 Type C are rare classes, so macro-F1 must be read with support and subgroup/error analysis.

New workflow:
- train_v74.py supports --single_task_target Task1|Task2|Task3|all.
- train_v74.py supports --locked_split_manifest and locked-test no-support output.
- error_analysis_v74.py writes rule_model_conflicts.csv and rule_model_conflict_summary.csv.
- sync_all_outputs_v74.py syncs selected formal source files and outputs into all/.

============================================================
9. 2026-06-11 missing-aware rule baseline
============================================================

Clinical rule baseline now reports both forced prediction and missing-aware abstain prediction.

Run:
python baseline_rules_v74.py --data-dir . --output-dir results_v74/rule_baselines_phase3

Outputs:
- rule_baseline_summary.csv/json
- rule_baseline_predictions.csv
- rule_baseline_evidence_summary.csv

Interpretation:
- forced_pred_label: rule still gives a label for every row.
- abstain_pred_label: rows with insufficient evidence become INSUFFICIENT_EVIDENCE.
- coverage: proportion of rows where the rule had complete/usable evidence.
- abstain_rate: proportion of rows where the rule should not be treated as a confident real baseline decision.
- Task2 keeps BC missing / no BC / NR / censored rows in n_total; they are not removed.

因此論文不建議主張「模型比臨床規則更準」。較合理的研究主軸是：
- missing / NR / censored audiological measurements 下的模型穩定性。
- ear-level multi-task learning 是否能在少樣本 Task2/Task3 中提供輔助。
- configured validation 與 deployment no-support 的差異。
- rule confidence / reward confidence 如何避免規則過度主導 CE loss。

## 2026-06-21 P0-P2 Robustness And Explainability Update
??????????

```powershell
python run_all_v74.py
```

??????????? compile?grouped locked-test split?training?clinical rule baseline?classical ML baseline??? error analysis?paper table export?no-support/locked-test hybrid evaluation?calibration?artificial missingness robustness?feature-group importance/ablation?model/deployment profile?all/ ????????

?? edge ? latency/profile ?????????`model_profile_v74.py` ?????????`run_all_v74.py` ????? `model_profile.csv`??????? edge/deployment summary???? `--require-edge-profile`?

?????????
- `results_v74/artificial_missingness/artificial_missingness_degradation_summary.csv`????????? complete data ??????
- `results_v74/artificial_missingness/evidence_compensation_summary.csv`????????????????????
- `results_v74/feature_importance/feature_group_permutation_importance.csv`?feature group permutation importance?
- `results_v74/feature_importance/feature_group_ablation_summary.csv`?inference-time feature group mean-mask ablation?
- hybrid evaluation ??? `hybrid_rule_first_confidence_gate`?? rule ??????????????? `INSUFFICIENT_EVIDENCE`?
- `sync_all_outputs_v74.py --clean` ????? `.pth/.pt/.ckpt`?????? GPT ??? all/ package?

## 2026-06-21 RUN_CONFIGS 15 + ABLATION 5 Masking Update
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
