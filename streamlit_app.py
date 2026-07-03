#==Imports ======================================================================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import subprocess

from src.train import main as retrain
from src.predict import predict, load_models, preprocess_input, align_column

#==Configuration=================================================================
st.set_page_config(
    page_title = "Loan Rsk Analysis",
    page_icon = "🏦",
    layout = "wide"
)

#==Side Bar Navigation===========================================================
st.sidebar.title("🏦 Loan Risk Analysis!")
page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Explaination", "Retrain Model"]
)

#==Page 1: Prediction ============================================================
if page == "prediction":
    st.title("Loan Default prediction")
    st.markdown("Fill in the applicant details below to predict loan default risk.")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=80, value=25)
        person_income = st.number_input("Person Income", min_value=1000, max_value=1000000, value=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=40.0, value=3.0)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])


    with col2:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=35000, value=10000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.5)
        loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=0.8, value=0.2)
        cb_person_default_on_file = st.selectbox("Previos Default on File", ["Y", "N"])
        cb_person_cred_hist_length = st.number_input("Credict History Length (years)", min_value=0.0, max_value=30, value=4)

    if st.button("Predict", type="primary"):
        input_dict = {
            "person_age" : person_age,
            "person_income" : person_income,
            "person_emp_length" : person_emp_length,
            "person_home_ownership" : person_home_ownership,
            "loan_intent" : loan_intent,
            "loan_grade" : loan_grade,
            "loan_amnt" : loan_amnt,
            "loan_int_rate" : loan_int_rate,
            "loan_percent_income" : loan_percent_income,
            "cb_person_default_on_file" : cb_person_default_on_file,
            "cb_person_cred_hist_length" : cb_person_cred_hist_length
        }

        result = predict(input_dict)

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if result['prediction'] == 'Default':
                st.error(f"🚨 {result['prediction']}")
            else:
                st.success(f"{result['prediction']}")
        
        with col_b:
            st.metric("Default Probability", f"{result['probability']*100:.2f}%")

        with col_c:
            st.metric("Threshold Used", f"{result['threshold_used']}")

#==Page 2: Explaination========================================================
elif page == 'Explaination':
    st.title("Shape Feature Importance")
    st.markdown("These plots explain how the model makes decisions.")

    tab1, tab2, tab3 = st.tabs(["Summery Plot", "Bar Plot", "Waterfall Plot"])

    with tab1:
        st.image("explainability/summery_plot.png", use_column_width=True)
        st.caption("Each dot represents one prediction. Red = high feature value, Blue = low feature value.")

    with tab2:
        st.image("explainability/bar_plot.png", use_column_width=True)
        st.caption("Average impact of each feature across all predictions.")
    
    with tab3:
        st.image("explainability/waterfall_plot.png", use_column_width=True)
        st.caption("How each feature pushed the prediction up or down for a single applicant.")

#==Page : Retrain==============================================================
elif page == "Retrain Model":
    st.title("Retrain Model")
    st.markdown("Click the button below to retrain the model with the latest data.")

    st.warning("⚠️ Retraining will overwrite the existing model. This may take a few minutes.")

    if st.button("Start Retraining", type='primary'):
        with st.spinner("Retraining in progress..."):
            try:
                result = subprocess.run(
                    ["python", "src/train.py"],
                    capture_output=True,
                    text=True
                )
                st.success("✅ Model retrained successfully!")
                st.code(result.stdout)
            except Exception as e:
                st.error(f"❌ Error: {e}")
