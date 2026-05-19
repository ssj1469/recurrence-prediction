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
        
        if hasattr(preprocessor, "get_feature_names_out") and hasattr(model, "feature_names_in_"):
         
            processed_cols = preprocessor.get_feature_names_out()
            x_processed_df = pd.DataFrame(x_processed, columns=processed_cols)
            
       
            x_final = x_processed_df[model.feature_names_in_]
        else:
         
            x_final = x_processed
            
    
        prob = model.predict_proba(x_final)[0,1]
        pred = int(prob >= 0.5)
        
        st.subheader("Prediction result")
        st.metric("Probability of recurrence", f"{prob:.3f}")
        st.write("Predicted class:", "**Recurred (1)**" if pred==1 else "**Non-recurred (0)**")
        st.caption("Note: This tool is for research/demo only. Clinical decisions must rely on professional judgement.")
        
    except Exception as e:
        st.error("🚨 侦察兵拦截到了真实的底层报错！请把下面这三条情报截图发给我：")
        

        st.code(f"错误类型: {type(e).__name__}\n详细原因: {str(e)}")

        if hasattr(model, 'feature_names_in_'):
            st.info(f"👉 模型点名要的 29 个特征 (前 8 个): {list(model.feature_names_in_[:8])}")
            
        if 'processed_cols' in locals():
            st.info(f"👉 翻译官给出的 55 个特征 (前 8 个): {list(processed_cols[:8])}")
