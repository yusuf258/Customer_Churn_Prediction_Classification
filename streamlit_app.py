import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page Config
st.set_page_config(page_title="Müşteri Kayıp Analizi", page_icon="📉", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'churn.csv')
ML_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
DL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dl_model.keras')

# --- DYNAMIC ASSET PREPARATION ---
@st.cache_resource
def prepare_assets_from_csv():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None
    
    # Load and Clean (Matching Notebook exactly)
    df = pd.read_csv(DATA_PATH)
    # Standardize column names: lowercase and remove special chars
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
    
    clean_df = df.drop('customerid', axis=1)
    
    # Pre-process TotalCharges (Coerce to numeric)
    clean_df['totalcharges'] = pd.to_numeric(clean_df['totalcharges'], errors='coerce')
    clean_df = clean_df.fillna(0)
    
    # Fit LabelEncoders on non-numeric columns
    encoders = {}
    for column in clean_df.columns:
        if column == 'churn': continue
        if not np.issubdtype(clean_df[column].dtype, np.number):
            le = LabelEncoder()
            clean_df[column] = le.fit_transform(clean_df[column].astype(str))
            encoders[column] = le
            
    # Target Encoder
    y_le = LabelEncoder()
    clean_df['churn'] = y_le.fit_transform(clean_df['churn'].astype(str))
    
    # Fit Scaler
    X = clean_df.drop('churn', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    
    return encoders, scaler, X.columns.tolist(), df # df has lowercased columns here

@st.cache_resource
def load_models():
    ml_model = joblib.load(ML_MODEL_PATH) if os.path.exists(ML_MODEL_PATH) else None
    dl_model = tf.keras.models.load_model(DL_MODEL_PATH) if os.path.exists(DL_MODEL_PATH) else None
    return ml_model, dl_model

# --- INITIALIZE ---
st.title("📉 Müşteri Kayıp (Churn) Tahmini")
st.markdown("Veri seti üzerinden anlık öğrenilen (fitting) encoder ve scaler ile tahminleme.")

encoders, scaler, feature_cols, raw_df = prepare_assets_from_csv()
ml_model, dl_model = load_models()

if encoders is not None and ml_model is not None:
    # Helper to get options from lowercased column names
    def get_options(col_name):
        return sorted(raw_df[col_name].unique().tolist())

    # Form
    with st.form("churn_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", get_options("gender"))
            seniorcitizen = st.selectbox("SeniorCitizen", [0, 1])
            partner = st.selectbox("Partner", get_options("partner"))
            dependents = st.selectbox("Dependents", get_options("dependents"))
            tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
            phoneservice = st.selectbox("PhoneService", get_options("phoneservice"))

        with c2:
            multiplelines = st.selectbox("MultipleLines", get_options("multiplelines"))
            internetservice = st.selectbox("InternetService", get_options("internetservice"))
            onlinesecurity = st.selectbox("OnlineSecurity", get_options("onlinesecurity"))
            onlinebackup = st.selectbox("OnlineBackup", get_options("onlinebackup"))
            deviceprotection = st.selectbox("DeviceProtection", get_options("deviceprotection"))
            techsupport = st.selectbox("TechSupport", get_options("techsupport"))

        with c3:
            streamingtv = st.selectbox("StreamingTV", get_options("streamingtv"))
            streamingmovies = st.selectbox("StreamingMovies", get_options("streamingmovies"))
            contract = st.selectbox("Contract", get_options("contract"))
            paperlessbilling = st.selectbox("PaperlessBilling", get_options("paperlessbilling"))
            paymentmethod = st.selectbox("PaymentMethod", get_options("paymentmethod"))
            monthlycharges = st.number_input("MonthlyCharges ($)", min_value=0.0, value=50.0)
            totalcharges = st.number_input("TotalCharges ($)", min_value=0.0, value=500.0)

        submitted = st.form_submit_button("Analiz Et ve Tahminle")

    if submitted:
        # 1. Create Input DF (Using lowercased keys to match feature_cols)
        input_dict = {
            'gender': gender, 'seniorcitizen': seniorcitizen, 'partner': partner,
            'dependents': dependents, 'tenure': tenure, 'phoneservice': phoneservice,
            'multiplelines': multiplelines, 'internetservice': internetservice,
            'onlinesecurity': onlinesecurity, 'onlinebackup': onlinebackup,
            'deviceprotection': deviceprotection, 'techsupport': techsupport,
            'streamingtv': streamingtv, 'streamingmovies': streamingmovies,
            'contract': contract, 'paperlessbilling': paperlessbilling,
            'paymentmethod': paymentmethod, 'monthlycharges': monthlycharges,
            'totalcharges': totalcharges
        }
        
        input_df = pd.DataFrame([input_dict])

        # 2. Encode
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # 3. Scale (Ensure column order matches feature_cols)
        X_input = input_df[feature_cols]
        X_scaled = scaler.transform(X_input)

        # 4. Predict
        res1, res2 = st.columns(2)
        
        with res1:
            st.subheader("🤖 ML (Random Forest)")
            prob = ml_model.predict_proba(X_scaled)[0][1]
            st.metric("Kayıp İhtimali", f"%{prob*100:.2f}")
            if prob > 0.5: st.error("RİSK: Müşteri Terk Edebilir")
            else: st.success("GÜVENLİ: Müşteri Kalıcı")

        if dl_model:
            with res2:
                st.subheader("🧠 DL (Neural Network)")
                prob_dl = dl_model.predict(X_scaled, verbose=0)[0][0]
                st.metric("Kayıp İhtimali", f"%{prob_dl*100:.2f}")
                if prob_dl > 0.5: st.error("RİSK: Müşteri Terk Edebilir")
                else: st.success("GÜVENLİ: Müşteri Kalıcı")
else:
    st.error("Veri dosyası (churn.csv) veya modeller bulunamadı.")
