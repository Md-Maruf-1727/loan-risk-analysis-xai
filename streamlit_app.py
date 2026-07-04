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

#==Custom Styling (Fonts, Colors, Animations)====================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

:root {
    --primary: #6C63FF;
    --primary-dark: #4B3FE0;
    --accent: #00D4B4;
    --danger: #FF5C7A;
    --bg-dark: #0F1120;
    --bg-card: #171A2B;
    --text-light: #EAEAF5;
    --text-muted: #9A9CB8;
}

.stApp {
    background: linear-gradient(135deg, #0F1120 0%, #161A2E 50%, #0F1120 100%);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color: var(--text-light);
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ---------- Fade / Slide in ---------- */
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes pulseGlow {
    0% {box-shadow: 0 0 0 0 rgba(108, 99, 255, 0.45);}
    70% {box-shadow: 0 0 0 14px rgba(108, 99, 255, 0);}
    100% {box-shadow: 0 0 0 0 rgba(108, 99, 255, 0);}
}
@keyframes shimmer {
    0% {background-position: -400px 0;}
    100% {background-position: 400px 0;}
}

.block-container {
    animation: fadeIn 0.6s ease-in-out;
    padding-top: 2rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14172A 0%, #0D0F1C 100%);
    border-right: 1px solid rgba(108, 99, 255, 0.25);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
    transition: all 0.25s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent) !important;
    transform: translateX(4px);
}
section[data-testid="stSidebar"] h1 {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    animation: fadeInUp 0.8s ease;
}

/* ---------- Headings ---------- */
h1, h2, h3 {
    font-weight: 700 !important;
    animation: fadeInUp 0.7s ease;
}
h1 {
    background: linear-gradient(90deg, #FFFFFF 20%, var(--primary) 60%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

/* ---------- Cards (columns / containers) ---------- */
div[data-testid="column"] {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 22px 20px;
    margin: 6px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeInUp 0.6s ease;
}
div[data-testid="column"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(108, 99, 255, 0.18);
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"], div[data-baseweb="select"] {
    border-radius: 10px !important;
    transition: all 0.25s ease;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
    box-shadow: 0 0 0 2px var(--primary) !important;
}
label, .stMarkdown p {
    color: var(--text-muted) !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    animation: pulseGlow 2.4s infinite;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 22px rgba(108, 99, 255, 0.4);
    filter: brightness(1.08);
}
.stButton > button:active {
    transform: translateY(0px) scale(0.98);
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1B1E33, #14172A);
    border-radius: 14px;
    padding: 14px 10px;
    border: 1px solid rgba(0, 212, 180, 0.2);
    animation: fadeInUp 0.8s ease;
    transition: transform 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.03);
}
div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-weight: 700 !important;
}

/* ---------- Alerts (success / error / warning) ---------- */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    animation: fadeInUp 0.5s ease;
    font-weight: 500;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: var(--text-muted);
    transition: all 0.25s ease;
}
button[data-baseweb="tab"]:hover {
    color: var(--accent);
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

/* ---------- Images ---------- */
div[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 8px 26px rgba(0,0,0,0.35);
    transition: transform 0.35s ease;
    animation: fadeIn 1s ease;
}
div[data-testid="stImage"] img:hover {
    transform: scale(1.015);
}

/* ---------- Code blocks ---------- */
.stCodeBlock, code {
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 10px !important;
}

/* ---------- Divider ---------- */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    margin: 1.5rem 0;
}

/* ---------- Hero banner ---------- */
.hero-banner {
    padding: 28px 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(108,99,255,0.15), rgba(0,212,180,0.10));
    border: 1px solid rgba(108, 99, 255, 0.25);
    margin-bottom: 1.4rem;
    animation: fadeInUp 0.7s ease;
}
.hero-banner h1 { margin-bottom: 4px; }
.hero-sub { color: var(--text-muted); font-size: 15px; }

.section-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    background: rgba(0, 212, 180, 0.12);
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    animation: fadeIn 1s ease;
}
</style>
""", unsafe_allow_html=True)

#==Side Bar Navigation===========================================================
st.sidebar.title("🏦 Loan Risk Analysis!")
st.sidebar.markdown(
    "<p style='color:#9A9CB8; font-size:13px; margin-top:-10px;'>AI-powered credit risk intelligence</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Explaination", "Retrain Model"]
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<p style='color:#9A9CB8; font-size:12px;'>Model: Random Forest · SHAP explainability</p>",
    unsafe_allow_html=True
)

#==Page 1: Prediction ============================================================
if page == "Prediction":
    st.markdown("""
        <div class="hero-banner">
            <span class="section-tag">🏦 PREDICTION ENGINE</span>
            <h1>Loan Default Prediction</h1>
            <p class="hero-sub">Fill in the applicant details below to predict loan default risk.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Applicant Profile")
        person_age = st.number_input("Age", min_value=18, max_value=80, value=25)
        person_income = st.number_input("Person Income", min_value=1000, max_value=1000000, value=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=40.0, value=3.0)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])

    with col2:
        st.markdown("##### 💳 Loan Details")
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=35000, value=10000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.5)
        loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=0.8, value=0.2)
        cb_person_default_on_file = st.selectbox("Previos Default on File", ["Y", "N"])
        cb_person_cred_hist_length = st.number_input("Credict History Length (years)", min_value=1, max_value=30, value=4)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    if predict_clicked:
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

        with st.spinner("Analyzing applicant risk profile..."):
            result = predict(input_dict)

        st.markdown("---")
        st.markdown("##### 📊 Prediction Result")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if result['prediction'] == 'Default':
                st.error(f"🚨 {result['prediction']}")
            else:
                st.success(f"✅ {result['prediction']}")

        with col_b:
            st.metric("Default Probability", f"{result['probability']*100:.2f}%")

        with col_c:
            st.metric("Threshold Used", f"{result['threshold_used']}")

        if result['prediction'] == 'Default':
            st.snow()
        else:
            st.balloons()

#==Page 2: Explaination========================================================
elif page == 'Explaination':
    st.markdown("""
        <div class="hero-banner">
            <span class="section-tag">🧠 MODEL INTERPRETABILITY</span>
            <h1>SHAP Feature Importance</h1>
            <p class="hero-sub">These plots explain how the model makes decisions.</p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Summary Plot", "📊 Bar Plot", "💧 Waterfall Plot"])

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
    st.markdown("""
        <div class="hero-banner">
            <span class="section-tag">⚙️ MODEL OPERATIONS</span>
            <h1>Retrain Model</h1>
            <p class="hero-sub">Click the button below to retrain the model with the latest data.</p>
        </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ Retraining will overwrite the existing model. This may take a few minutes.")

    if st.button("🔁 Start Retraining", type='primary', use_container_width=True):
        with st.spinner("Retraining in progress..."):
            try:
                result = subprocess.run(
                    ["python", "src/train.py"],
                    capture_output=True,
                    text=True
                )
                st.success("✅ Model retrained successfully!")
                st.balloons()
                st.code(result.stdout)
            except Exception as e:
                st.error(f"❌ Error: {e}")