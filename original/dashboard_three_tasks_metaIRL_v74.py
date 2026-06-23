import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch

from mtl_meta_irl_transformer import MTLMetaIRLTransformer

SUPPORTED_CC = {(5,0),(6,0),(6,1),(7,0),(7,5),(8,0),(8,6),(9,0)}


def safe_pick_device():
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) not in SUPPORTED_CC:
                st.warning(f"⚠ GPU sm_{major}{minor} 不支援目前 PyTorch → 自動切換 CPU")
                return torch.device("cpu")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


DEVICE = safe_pick_device()


@st.cache_resource
def load_model(path="results_v74/full_seed_0/best_model.pth"):
    candidates = [
        path,
        "checkpoints/MTL_MetaIRL_v73_best.pth",
        "checkpoints/MTL_MetaIRL_v72_best.pth",
        "checkpoints/MTL_MetaIRL_v71_best.pth",
        "best_model_metaIRL_v71.pth",
    ]
    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break
    if found is None:
        st.error(f"❌ 找不到 checkpoint，已檢查：{candidates}")
        return None, None
    if found != path:
        st.warning(f"⚠ 找不到 {path}，改用 {found}")

    ckpt = torch.load(found, map_location="cpu")
    union_features = ckpt["meta"]["union_features"]
    tasks_meta = ckpt["meta"]["tasks"]

    model = MTLMetaIRLTransformer(
        input_dim=len(union_features),
        task_label_class_counts={t: v["num_classes"] for t, v in tasks_meta.items()},
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.15,
        enable_meta=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, ckpt["meta"]


model, meta = load_model()
if model is None:
    st.stop()


def clean_value(v):
    try:
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def norm_row(task_name, row_dict):
    feats = meta["union_features"]
    mu = meta["norm_meta"][task_name]["mu"]
    sigma = meta["norm_meta"][task_name]["sigma"]
    out = []
    for f in feats:
        x = clean_value(row_dict.get(f, 0.0))
        out.append((x - mu[f]) / sigma[f])
    return np.array(out, dtype=np.float32)


def expert_consistency_score(task_name, label_col, row, pred_name):
    side = "right" if "right" in label_col.lower() else "left"
    p = str(pred_name).upper()
    if task_name == "Task1":
        pta = clean_value(row.get(f"{side}_PTA", 0.0))
        if pta < 20 and "NORMAL" in p: return 1.0
        if 20 <= pta < 35 and "MILD" in p: return 1.0
        if 35 <= pta < 50 and "MODERATE" in p: return 1.0
        if 50 <= pta < 65 and ("MODERATELY" in p or "SEVERE" in p): return 1.0
        if 65 <= pta < 80 and "SEVERE" in p: return 1.0
        if pta >= 80 and ("PROFOUND" in p or "COMPLETE" in p): return 1.0
        return -0.2
    elif task_name == "Task2":
        ac = np.mean([clean_value(row.get(f"{side}_500Hz", 0.0)),
                      clean_value(row.get(f"{side}_1000Hz", 0.0)),
                      clean_value(row.get(f"{side}_2000Hz", 0.0)),
                      clean_value(row.get(f"{side}_4000Hz", 0.0))])
        bc = np.mean([clean_value(row.get(f"bc_{side}_500Hz", 0.0)),
                      clean_value(row.get(f"bc_{side}_1000Hz", 0.0)),
                      clean_value(row.get(f"bc_{side}_2000Hz", 0.0)),
                      clean_value(row.get(f"bc_{side}_4000Hz", 0.0))])
        abg = ac - bc
        if abg >= 15 and ("CHL" in p or "MHL" in p or "MIHL" in p): return 1.0
        if abg < 10 and ("SNHL" in p or "WNL" in p or "TONE LOSS" in p): return 0.8
        return -0.2
    elif task_name == "Task3":
        peak = clean_value(row.get(f"tymp_{side}_peak_daPa", 0.0))
        cmpl = clean_value(row.get(f"tymp_{side}_peak_mmho", 0.0))
        width = clean_value(row.get(f"tymp_{side}_Width_daPa", 0.0))
        if abs(peak) <= 100 and 0.2 <= cmpl <= 1.8 and width <= 200 and p == "A": return 1.0
        if cmpl < 0.15 and p == "B": return 1.0
        if peak < -100 and p == "C": return 1.0
        return -0.2
    return 0.0


def predict_single(row_dict, task_name):
    x = norm_row(task_name, row_dict)
    x = torch.tensor(x).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_dict, reward_dict = model(x, task_name, support=None)

    results = {}
    label_cols = meta["tasks"][task_name]["label_cols"]
    class_names = meta["tasks"][task_name]["class_names"]

    for lbl in label_cols:
        logits = logits_dict[lbl][0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[lbl][pred_idx]
        reward_pred = float(reward_dict[lbl][0].cpu().item())
        expert = expert_consistency_score(task_name, lbl, row_dict, pred_name)
        results[lbl] = {
            "pred": pred_name,
            "probs": probs,
            "reward_pred": reward_pred,
            "expert_consistency": expert,
            "class_names": class_names[lbl],
        }
    return results


def batch_predict(df, task_name):
    outs, errs = [], []
    for idx, row in df.iterrows():
        try:
            res = predict_single(row.to_dict(), task_name)
            row_out = {"index": idx}
            for lbl, info in res.items():
                row_out[f"{lbl}_pred"] = info["pred"]
                row_out[f"{lbl}_reward"] = info["reward_pred"]
                row_out[f"{lbl}_expert_consistency"] = info["expert_consistency"]
                for i, cname in enumerate(info["class_names"]):
                    row_out[f"{lbl}_prob_{cname}"] = info["probs"][i]
            outs.append(row_out)
        except Exception as e:
            errs.append({"index": idx, "error": str(e)})
    return pd.DataFrame(outs), pd.DataFrame(errs)


st.set_page_config(page_title="MetaIRL Hearing AI v7.4", layout="wide")
st.title("🎧 Meta-Learning + Offline-IRL Hearing AI v7.4")
st.caption("三任務雙耳 MTL：WHO degree / hearing type / tympanogram type")
st.info("v7.4 對齊 v7.3 資料清理流程，並支援多 seed / ablation 的研究版 checkpoint。")


def side_num(label, default=0.0):
    return st.number_input(label, value=default, step=0.1)


with st.sidebar:
    st.subheader("輸入雙耳 audiogram / BC / tymp")
    vals = {}
    vals["right_500Hz"] = side_num("右耳 500Hz")
    vals["right_1000Hz"] = side_num("右耳 1000Hz")
    vals["right_2000Hz"] = side_num("右耳 2000Hz")
    vals["right_4000Hz"] = side_num("右耳 4000Hz")
    vals["right_PTA"] = side_num("右 PTA")
    vals["left_500Hz"] = side_num("左耳 500Hz")
    vals["left_1000Hz"] = side_num("左耳 1000Hz")
    vals["left_2000Hz"] = side_num("左耳 2000Hz")
    vals["left_4000Hz"] = side_num("左耳 4000Hz")
    vals["left_PTA"] = side_num("左 PTA")
    vals["bc_right_500Hz"] = side_num("右 BC 500Hz")
    vals["bc_right_1000Hz"] = side_num("右 BC 1000Hz")
    vals["bc_right_2000Hz"] = side_num("右 BC 2000Hz")
    vals["bc_right_4000Hz"] = side_num("右 BC 4000Hz")
    vals["bc_left_500Hz"] = side_num("左 BC 500Hz")
    vals["bc_left_1000Hz"] = side_num("左 BC 1000Hz")
    vals["bc_left_2000Hz"] = side_num("左 BC 2000Hz")
    vals["bc_left_4000Hz"] = side_num("左 BC 4000Hz")
    vals["tymp_right_Vea"] = side_num("右 Vea")
    vals["tymp_right_peak_daPa"] = side_num("右 peak daPa")
    vals["tymp_right_peak_mmho"] = side_num("右 peak mmho")
    vals["tymp_right_Width_daPa"] = side_num("右 Width daPa")
    vals["tymp_left_Vea"] = side_num("左 Vea")
    vals["tymp_left_peak_daPa"] = side_num("左 peak daPa")
    vals["tymp_left_peak_mmho"] = side_num("左 peak mmho")
    vals["tymp_left_Width_daPa"] = side_num("左 Width daPa")

tabs = st.tabs(["Task1 WHO", "Task2 Type", "Task3 Tymp"])
for task_name, tab in zip(["Task1", "Task2", "Task3"], tabs):
    with tab:
        st.subheader(task_name)
        if st.button(f"▶ 單筆推論 {task_name}", key=f"single_{task_name}"):
            out = predict_single(vals, task_name)
            for lbl, info in out.items():
                st.markdown(f"### {lbl}")
                st.write(f"**Prediction:** {info['pred']}")
                st.write(f"Reward score: `{info['reward_pred']:.3f}`")
                st.write(f"Clinical consistency prior: `{info['expert_consistency']:.3f}`")
                chart_df = pd.DataFrame({"class": info["class_names"], "probability": info["probs"]})
                st.bar_chart(chart_df.set_index("class"))
        st.write("---")
        upload = st.file_uploader(f"上傳 {task_name} CSV", type=["csv"], key=f"file_{task_name}")
        if upload and st.button(f"▶ Batch 推論 {task_name}", key=f"batch_{task_name}"):
            df = pd.read_csv(upload)
            preds, errors = batch_predict(df, task_name)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"MetaIRL_v74_{task_name}_{now}.xlsx"
            with pd.ExcelWriter(out_path) as writer:
                preds.to_excel(writer, sheet_name="predictions", index=False)
                errors.to_excel(writer, sheet_name="errors", index=False)
            st.success(f"✅ 匯出完成：{out_path}")
            with open(out_path, "rb") as f:
                st.download_button("📥 下載 Excel 結果", data=f.read(), file_name=out_path)
