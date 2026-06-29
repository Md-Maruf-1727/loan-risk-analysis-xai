#--Imports---------------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import pipeline 

#--Config------------------------------------------------------------------------

DATA_PATH = "Data/credit_risk_dataset.csv"
MODEL_PATH = "models/rf_model.joblib"
FEATURE_PATH = "models/feature_columns.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLD = 5
TARGET = 'loan_status'

#--Load Data --------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

#--Data Cleaning -----------------------------------------------------------------
def clean_data(df):
    #remove duplicate
    df = df.drop_duplicates()

    #Outlier Remove
    df = df[df['person_age'] < 80]
    df = df[df['person_emp_length'] < 32]
    df = df[df['person_income'] < 1000000]
    df = df[df['loan_percent_income'] < 0.8]

    #Missing Value Fill
    df['loan_percent_income'] = df['loan_percent_income'].fillna(df['loan_percent_income'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    return df

#--Encode Data------------------------------------------------------------------
def encode_data(df):
    #Onehot Encoding
    person_home_ownership_dummy = pd.get_dummies(
        df['person_home_ownership'],
        prefix='person_home_ownership',
        drop_first=True
    ).astype('int')

    loan_intent_dummy = pd.get_dummies(
        df['loan_intent'],
        prefix='loan_intent',
        drop_first=True
    ).astype('int')

    df = pd.concat([df, person_home_ownership_dummy, loan_intent_dummy], axis=1)
    df = df.drop(['person_home_ownership', 'loan_intent'], axis=1)

    #Ordinal Encoding
    grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
    df['loan_grade'] = df['loan_grade'].map(grade_map)

    #Label Encoding
    cb_map = {'Y':1, 'N':0}
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(cb_map)

    return df

#--Feature Engineering--------------------------------------------------------
def feature_engineering(df):
    df.insert(
        loc=2, column='person_monthly_income',
        value=(df['person_income'] / 12).round(2)
    )
    df.insert(
        loc=4, column='emp_stability',
        value=(df['person_emp_length'] / df['person_age']).round(2)
    )
    df.insert(
        loc=13, column='credit_hist_ratio',
        value=(df['cb_person_cred_hist_length'] / df['person_age']).round(2)
    )
    df.insert(
        loc=14, column='debt_burden_index',
        value=((df['loan_amnt'] * (1 + df['loan_int_rate'] / 100)) / df['person_income']).round(2)
    )

    return df

#--Log Transformation---------------------------------------------------------
def log_transform(df):
    num_cols = [
    'person_age', 'person_income', 
    'person_monthly_income', 'emp_stability',
    'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 
    'debt_burden_index', 'cb_person_cred_hist_length'
    ]

    for col in num_cols:
        df[col] = np.log1p(df[col])
    
    return df

#--Split Data ---------------------------------------------------------------
def split_data(df):
    x = df.drop(TARGET, axis=1)
    y = df[TARGET]

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y,
        random_state=RANDOM_STATE,
        stratify=y, test_size=TEST_SIZE
    )
    return xtrain, xtest, ytrain, ytest

#--Train Model ---------------------------------------------------------------
def train_model(xtrain, ytrain):
    pipeline = pipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    
    rf_param_dist = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 15],
        'model__min_samples_split': [2, 5, 10]
    }

    rf_cv = StratifiedKFold(
        n_splits=CV_FOLD, 
        shuffle= True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator= pipeline,
        param_distributions= rf_param_dist,
        scoring='recall',
        cv=rf_cv, n_iter=10,
        verbose=1, n_jobs=-1
    )

    search.fit(xtrain, ytrain) 
    return search

    