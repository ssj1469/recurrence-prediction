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

    x_raw = pd.DataFrame([inputs])
    
   
    ultimate_pipe = joblib.load("ultimate_pipe_ds1.joblib")
    

    prob = ultimate_pipe.predict_proba(x_raw)[0,1]
    pred = int(prob >= 0.5)
    
    st.subheader("Prediction result")
    st.metric("Probability of recurrence", f"{prob:.3f}")
    st.write("Predicted class:", "**Recurred (1)**" if pred==1 else "**Non-recurred (0)**")
    st.caption("Note: This tool is for research/demo only. Clinical decisions must rely on professional judgement.")
    

    try:
        st.subheader("SHAP Explanation")
        
        
        actual_model = ultimate_pipe.named_steps['classifier']

        x_processed_29 = ultimate_pipe[:-1].transform(x_raw)
        
     
        feature_names_29 = ultimate_pipe[:-1].get_feature_names_out()
        x_shap_df = pd.DataFrame(x_processed_29, columns=feature_names_29)
        
        
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer(x_shap_df)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"SHAP 图生成遇到小问题，但不影响上面的预测结果！报错信息：{e}")
