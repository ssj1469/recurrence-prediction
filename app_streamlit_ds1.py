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
        
  
        # 获取 SHAP 背景数据的所有列名（并剔除可能生成的空序号列）
        bg_cols = [c for c in bg.columns if 'Unnamed' not in c]
        
     
        try:
          
            x_final = x_processed_df[bg_cols]
        except KeyError:
          
            clean_cols = [c.split('__')[-1] if '__' in c else c for c in cols_55]
            x_processed_df.columns = clean_cols
            x_final = x_processed_df[bg_cols]
            

        prob = model.predict_proba(x_final)[0,1]
        pred = int(prob >= 0.5)
        
        st.subheader("Prediction result")
        st.metric("Probability of recurrence", f"{prob:.3f}")
        st.write("Predicted class:", "**Recurred (1)**" if pred==1 else "**Non-recurred (0)**")
        st.caption("Note: This tool is for research/demo only. Clinical decisions must rely on professional judgement.")
        

        
    except Exception as e:
   
        st.error("🚨 最后的挣扎：对齐失败！请截图发给我：")
        st.info(f"大夫想要的特征 (bg_sample 里的前 8 个): {bg_cols[:8]}")
        st.info(f"翻译官给出的特征 (去前缀后的前 8 个): {list(x_processed_df.columns)[:8]}")
