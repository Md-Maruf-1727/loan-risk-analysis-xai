# 🏦 Loan Risk Analysis with Explainable AI (XAI)

A machine learning system that predicts loan default risk and explains *why* each prediction was made, using SHAP-based explainability and an interactive Streamlit dashboard.

🔗 **Live Demo:** [loan-risk-analysis-xai.streamlit.app](https://loan-risk-analysis-xai-ucuhrzekyxrpvkfahnsbsq.streamlit.app/)

---

## 📌 Overview

Financial institutions need not only accurate default-risk predictions but also transparent, interpretable reasoning behind them. This project builds an end-to-end pipeline — from raw credit data to a deployed, explainable prediction app — that:

- Cleans and preprocesses a real-world credit risk dataset
- Engineers domain-relevant financial features
- Trains and compares multiple classification models
- Selects the best model based on business-relevant metrics (recall-focused, since missing a defaulter is costlier than a false alarm)
- Explains predictions at both the global and individual level using SHAP
- Serves everything through an interactive Streamlit web app

---

## 🗂️ Project Structure

```
loan-risk-analysis-xai/
│
├── Data/
│   ├── credit_risk_dataset.csv       # Raw dataset
│   ├── Cleaned_dataset.csv           # After cleaning & outlier removal
│   └── Preprocessed_data.csv         # After encoding & feature engineering
│
├── Notebooks/
│   ├── 01_data_training.ipynb        # EDA, cleaning, outlier handling
│   ├── 02_data_preprocessing.ipynb   # Encoding, feature engineering, transforms
│   ├── 04_model_training.ipynb       # Model training, tuning & evaluation
│   └── 05_explainability.ipynb       # SHAP-based model explainability
│
├── explainability/
│   ├── summery_plot.png              # SHAP summary plot
│   ├── bar_plot.png                  # SHAP feature importance (bar)
│   └── waterfall_plot.png            # SHAP waterfall (single prediction)
│
├── models/
│   ├── rf_model.joblib               # Final trained Random Forest model
│   ├── feature_columns.joblib        # Saved feature column order
│   └── threshold.joblib              # Optimal classification threshold
│
├── src/
│   ├── train.py                      # Reusable training pipeline
│   └── predict.py                    # Preprocessing + inference pipeline
│
├── streamlit_app.py                  # Web app (Prediction, Explanation, Retrain)
└── requirements.txt                  # Project dependencies
```

---

## 📊 Dataset

- **Source:** Credit Risk Dataset (`credit_risk_dataset.csv`)
- **Size:** ~32,500 rows × 12 columns
- **Target variable:** `loan_status` (`0` = paid, `1` = default)
- **Class balance:** ~78% no-default vs ~22% default (handled with SMOTE)

**Key raw features:** age, income, employment length, home ownership, loan intent, loan grade, loan amount, interest rate, loan-to-income ratio, prior default history, and credit history length.

---

## 🧹 Data Cleaning

- Removed duplicate rows
- Removed unrealistic outliers (age > 80, employment length > 40 years, income > 1,000,000, loan-to-income ratio > 0.8)
- Filled missing values (`person_emp_length`, `loan_int_rate`) using the median

## 🛠️ Feature Engineering

New features created to better capture financial risk signals:

| Feature | Description |
|---|---|
| `person_monthly_income` | Monthly income derived from annual income |
| `emp_stability` | Employment length relative to age |
| `credit_hist_ratio` | Credit history length relative to age |
| `debt_burden_index` | Loan amount adjusted for interest rate, relative to income |

**Encoding:**
- One-hot encoding → `person_home_ownership`, `loan_intent`
- Ordinal encoding → `loan_grade` (A–G)
- Label encoding → `cb_person_default_on_file`

**Transformation:** Log transformation (`log1p`) applied to skewed numerical columns to reduce the effect of outliers.

---

## 🤖 Modeling

Three algorithms were trained and compared:

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | ~86% | ~53% | ~0.62 | — |
| XGBoost Classifier | ~89% | ~78% | ~0.75 | 0.92 |
| **Random Forest (final)** | **~93%** | **~72–76%** | **~0.82–0.89** | **0.93** |

**Approach:**
- `StratifiedKFold` cross-validation (5 folds)
- `RandomizedSearchCV` for hyperparameter tuning
- `SMOTE` oversampling inside the pipeline to handle class imbalance
- Recall-optimized scoring, since correctly catching defaulters matters more than overall accuracy
- **Threshold tuning** via precision-recall curve to find the F1-optimal decision threshold (rather than the default 0.5)

**Final model:** Random Forest Classifier — chosen for its best balance of precision, recall, and ROC-AUC after threshold tuning.

---

## 🔍 Explainability (XAI)

Model decisions are explained using **SHAP (SHapley Additive exPlanations)**:

- **Summary Plot** — shows overall feature impact and direction across all predictions
- **Bar Plot** — ranks features by average importance
- **Waterfall Plot** — breaks down exactly how each feature pushed a single prediction toward default / no-default

This makes the model's reasoning transparent for both developers and non-technical stakeholders (e.g. loan officers).

---

## 🌐 Streamlit App

Try it live: **[loan-risk-analysis-xai.streamlit.app](https://loan-risk-analysis-xai-ucuhrzekyxrpvkfahnsbsq.streamlit.app/)**

The app (`streamlit_app.py`) has three pages:

1. **Prediction** — Enter applicant details and get an instant default prediction with probability and threshold used
2. **Explanation** — View SHAP summary, bar, and waterfall plots to understand model behavior
3. **Retrain Model** — Retrain the model on updated data directly from the UI

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Md-Maruf-1727/loan-risk-analysis-xai.git
   cd loan-risk-analysis-xai
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🧰 Tech Stack

- **Language:** Python
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Deployment:** Streamlit (Streamlit Community Cloud)
- **Model Persistence:** joblib

---

## 🚀 Future Improvements

- Add model monitoring / drift detection
- Expand explainability to LIME for cross-validation of SHAP insights
- Add automated unit tests for the preprocessing pipeline
- CI/CD pipeline for automatic redeployment on model updates

---

## 👤 Author

**Md. Maruf**
ML / Data Science Freelancer
GitHub: [Md-Maruf-1727](https://github.com/Md-Maruf-1727)
