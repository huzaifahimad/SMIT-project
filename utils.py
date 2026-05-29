import pandas as pd
import numpy as np
import joblib
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def load_data():
    df = pd.read_csv(os.path.join(BASE, 'data', 'Telco-Customer-Churn.csv'))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df

def load_metrics():
    with open(os.path.join(BASE, 'models', 'metrics.json'), 'r') as f:
        return json.load(f)

def load_champion():
    return joblib.load(os.path.join(BASE, 'models', 'champion_model.pkl'))

def load_scaler():
    return joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))

def load_feature_columns():
    return joblib.load(os.path.join(BASE, 'models', 'feature_columns.pkl'))

def preprocess_input(input_dict):
    """Preprocess a single prediction input dict into model-ready array."""
    binary_map = {'Yes':1,'No':0,'Male':1,'Female':0,
                  'No phone service':0,'No internet service':0}
    d = {}
    binary_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService',
                   'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in binary_cols:
        val = input_dict.get(col, 'No')
        if col == 'SeniorCitizen':
            d[col] = int(val)
        else:
            d[col] = binary_map.get(str(val), 0)

    d['tenure']         = float(input_dict.get('tenure', 12))
    d['MonthlyCharges'] = float(input_dict.get('MonthlyCharges', 65))
    d['TotalCharges']   = float(input_dict.get('TotalCharges', 780))

    internet = input_dict.get('InternetService','Fiber optic')
    d['InternetService_DSL']         = 1 if internet == 'DSL' else 0
    d['InternetService_Fiber optic'] = 1 if internet == 'Fiber optic' else 0
    d['InternetService_No']          = 1 if internet == 'No' else 0

    contract = input_dict.get('Contract','Month-to-month')
    d['Contract_Month-to-month'] = 1 if contract == 'Month-to-month' else 0
    d['Contract_One year']       = 1 if contract == 'One year' else 0
    d['Contract_Two year']       = 1 if contract == 'Two year' else 0

    payment = input_dict.get('PaymentMethod','Electronic check')
    d['PaymentMethod_Bank transfer (automatic)']  = 1 if payment == 'Bank transfer (automatic)' else 0
    d['PaymentMethod_Credit card (automatic)']    = 1 if payment == 'Credit card (automatic)' else 0
    d['PaymentMethod_Electronic check']           = 1 if payment == 'Electronic check' else 0
    d['PaymentMethod_Mailed check']               = 1 if payment == 'Mailed check' else 0

    scaler = load_scaler()
    feature_cols = load_feature_columns()

    df_input = pd.DataFrame([d])
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)
    df_input[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
        df_input[['tenure','MonthlyCharges','TotalCharges']])
    return df_input

