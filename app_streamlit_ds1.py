import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recurrence prediction (Dataset 1)", layout="wide")


model = joblib.load("final_pipe_ds1.joblib")
preprocessor = joblib.load("preprocessor_ds1.pkl")

with open("meta_ds1.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
bg = pd.read_csv("bg_sample_ds1.csv")

st.title("Recurrence prediction and SHAP explanation (Dataset 1)")
st.markdown("Please input patient features in the sidebar, then click the button to predict recurrence and view SHAP waterfall plot.")

with st.sidebar:
    st.header("Input features")
    inputs = {}
    for c in meta["num_cols"]:
        rng = meta["num_ranges"][c]
        val = st.number_input(c, float(rng["min"]), float(rng["max"]), float(rng["mean"]))
        inputs[c] = val
    for c in meta["cat_cols"]:
        options = meta["cat_values"].get(c, [])
        if len(options) == 0: val = st.text_input(c, "")
        else: val = st.selectbox(c, options, index=0)
        inputs[c] = val
    submit = st.button("Predict and explain")

if submit:
    try:
  
        x_raw = pd.DataFrame([inputs])
        x_processed = preprocessor.transform(x_raw)
        cols_55 = preprocessor.get_feature_names_out()
        x_processed_df = pd.DataFrame(x_processed, columns=cols_55)
        

        target_key = None
        for k, v in meta.items():
            # 寻找：必须是列表，长度必须刚好是 29，且名字能在这 55 列里找得到
            if isinstance(v, list) and len(v) == 29 and all(feat in cols_55 for feat in v):
                target_key = k
                break
        
        if target_key:
        
            x_final = x_processed_df[meta[target_key]]
        else:
           
            raise ValueError("糟糕！在 meta_ds1.json 中找到了特征，但名字和前缀对不上！")
            
      
        prob = model.predict_proba(x_final)[0,1]
        pred = int(prob >= 0.5)
        
        st.subheader("Prediction result")
        st.metric("Probability of recurrence", f"{prob:.3f}")
        st.write("Predicted class:", "**Recurred (1)**" if pred==1 else "**Non-recurred (0)**")
        st.caption("Note: This tool is for research/demo only. Clinical decisions must rely on professional judgement.")
        
    except Exception as e:
       
        st.error(f"🚨 侦察兵最终报告: {str(e)}")
        st.write("肯定是名字的前缀（比如 num__ 或 cat__）被去掉了！请把下面这个字典截图发给我，我一眼就能看出名单藏在哪个键里：")
        st.json({k: (f"包含 {len(v)} 个元素" if isinstance(v, list) else "其它") for k, v in meta.items()})
