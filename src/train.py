#--Imports-------------------------------------------

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import pipeline 

#--Config---------------------------------------------

DATA_PATH = "Data/credit_risk_dataset.csv"
MODEL_PATH = "models/rf_model.joblib"
FEATURE_PATH = "models/feature_columns.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLD = 5
TARGET = 'loan_status'

#--Load Data -------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

#--Data Cleaning ---------------------------------
def clean_data(df):
    #remove duplicate
    df = df.drop_duplicates()

    #

