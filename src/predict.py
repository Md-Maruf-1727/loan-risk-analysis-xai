#Import=======================================================================
import pandas as pd
import joblib
import os

try:
    from src.train import clean_data, encode_data, feature_engineering, log_transform
except ModuleNotFoundError:
    from train import clean_data, encode_data, feature_engineering, log_transform

#Configuration================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.joblib")

#Load saved model=============================================================
def load_models():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, feature_columns, threshold

#==Preprocess single input=====================================================
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    df = encode_data(df)
    df = feature_engineering(df)
    df = log_transform(df)

    return df

#==Align Column================================================================
def align_column(df, feature_columns):
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

#==predict model===============================================================
def predict(input_dict):
    model, feature_columns, threshold = load_models()

    df = preprocess_input(input_dict)
    df = align_column(df, feature_columns)

    probability = model.predict_proba(df)[:, 1][0]
    prediction = int(probability >= threshold)

    return {
        'prediction' : 'Default' if prediction == 1 else 'No Default',
        'probability': round(float(probability), 4),
        'threshold_used' : round(float(threshold), 4)
    }


#==Main block==================================================================
if __name__ == '__main__':

    sample_input = {
        'person_age' : 25,
        'person_income': 50000,
        'person_home_ownership' : 'RENT',
        'person_emp_length' : 3,
        'loan_intent' : 'EDUCATION',
        'loan_grade' : 'B',
        'loan_amnt' : 11.5,
        'loan_int_rate' : 11.5,
        'loan_percent_income' : 0.2,
        'cb_person_default_on_file' : 'N',
        'cb_person_cred_hist_length' : 4
    }

    result = predict(sample_input)

    print("Prediction Result!!")
    print(f"Prdiction     : {result['prediction']}")
    print(f"Probability   : {result['probability']}")
    print(f"Threshold Used: {result['threshold_used']}")