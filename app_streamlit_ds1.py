
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recurrence prediction (Dataset 1)", layout="wide")

ART_DIR = Path(r"D:\dataset2\dataset1\dataset1\20260306-数据集1结果图-01\artifacts")
model = joblib.load(ART_DIR / "final_pipe_ds1.joblib")
with open(ART_DIR / "meta_ds1.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
bg = pd.read_csv(ART_DIR / "bg_sample_ds1.csv")

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
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    prob = model.predict_proba(x)[0,1]
    pred = int(prob >= 0.5)
    st.subheader("Prediction result")
    st.metric("Probability of recurrence", f"{prob:.3f}")
    st.write("Predicted class:", "**Recurred (1)**" if pred==1 else "**Non-recurred (0)**")
    st.caption("Note: This tool is for research/demo only. Clinical decisions must rely on professional judgement.")
