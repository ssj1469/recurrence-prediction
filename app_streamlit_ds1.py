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
       
        st.error("🚨 还是遇到了特征维度不匹配的难题！")
        model_needs = getattr(model, 'n_features_in_', '未知')
        you_provided = x_processed.shape[1] if 'x_processed' in locals() else '未知'
        st.warning(f"情报侦测：你的模型需要 {model_needs} 列特征，但网页预处理输出了 {you_provided} 列。")
        st.write("请直接把上面这个黄框截图发给我，我们就知道具体差了几列！")
