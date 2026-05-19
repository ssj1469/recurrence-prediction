import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib, json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recurrence prediction (Dataset 1)", layout="wide")


ultimate_pipe = joblib.load("ultimate_pipe_ds1.joblib")

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
    
    prob = ultimate_pipe.predict_proba(x_raw)[0,1]
    pred = int(prob >= 0.5)
    
    st.subheader("Prediction result")
    st.metric("Probability of recurrence", f"{prob:.3f}")
    st.write("Predicted class:", "**Recurred (1)**" if pred==1 else "**Non-recurred (0)**")
    st.caption("Note: This tool is for research/demo only. Clinical decisions must rely on professional judgement.")
    
    try:
        st.subheader("SHAP Explanation")
        
        preprocessor_step = ultimate_pipe[0]
        selector_step = ultimate_pipe[1]
        classifier_step = ultimate_pipe[-1]
        
        x_processed_final = ultimate_pipe[:-1].transform(x_raw)
        
        cols_all = np.array(preprocessor_step.get_feature_names_out())
        support_mask = selector_step.support_
        feature_names_selected = cols_all[support_mask]
        
        x_shap_df = pd.DataFrame(x_processed_final, columns=feature_names_selected)
        explainer = shap.TreeExplainer(classifier_step)
        shap_values = explainer(x_shap_df)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"SHAP Plotting Error: {e}")
