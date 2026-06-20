# 🏦 Loan Risk Analysis (XAI) — Work In Progress

> ⚠️ **This project is currently in progress.**
> Data cleaning and preprocessing are complete.
> Model training and Explainable AI (XAI) implementation coming soon.

---

## ✅ Completed So Far

**Data Cleaning:**
- Removed 165 duplicate rows
- Removed unrealistic ages (age > 80) and employment lengths (> 40 years)
- Filled missing values using median imputation
- Removed extreme outliers in income, employment length, and loan percent income

**Data Preprocessing:**
- One-hot encoding for `person_home_ownership` and `loan_intent`
- Ordinal encoding for `loan_grade` (A=0 to G=6)
- Label encoding for `cb_person_default_on_file`
- Created 6 new engineered features:
  - `person_monthly_income`
  - `emp_stability` (employment length / age)
  - `interest_loan_ratio`
  - `total_debt_load`
  - `credit_hist_ratio`
  - `debt_burden_index`
- Applied log transformation to reduce skewness in numerical columns

---

## 📊 Dataset

- **Source:** Credit Risk Dataset
- **Original Size:** 32,581 rows × 12 columns
- **Target:** `loan_status` (0 = paid, 1 = default)
- **Class Imbalance:** 78.2% paid vs 21.8% default — will be handled in modeling

---

## 🔜 Coming Soon

- Exploratory Data Analysis (EDA)
- Model Training (XGBoost, Random Forest)
- Explainable AI (SHAP values)
- Feature importance visualization

---

## 👤 Author

**Md. Maruf**
GitHub: [github.com/Md-Maruf-1727](https://github.com/Md-Maruf-1727)
