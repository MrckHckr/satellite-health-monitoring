# IMPORT
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter

# CONFIG STREAMLIT
st.set_page_config(
    page_title="Satellite Health Monitoring",
    layout="wide"
)

st.title("🛰️ Satellite Health Monitoring Dashboard")

# LOAD MODELLI E SCALER
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    return lstm, scaler

lstm_model, scaler = load_models()

# LOAD DATI
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/telemetry.csv")

df = load_data()

# INFERENZA
# Preparazione finestra LSTM
WINDOW = 30

X = scaler.transform(
    df[["temperature", "vibration", "power"]]
)

X_seq = np.array([
    X[i:i+WINDOW]
    for i in range(len(X) - WINDOW)
])

# Probabilità di failure
failure_prob = lstm_model.predict(X_seq, verbose=0)

# Decision support
df = df.iloc[WINDOW:].copy()
df["failure_probability"] = failure_prob.flatten()

# VISUALIZZAZIONE + ALERT
# Grafici
st.subheader("📈 Failure Probability")

st.line_chart(
    df.set_index("timestamp")["failure_probability"]
)

# Alert operativi
if df["failure_probability"].iloc[-1] > 0.8:
    st.error("🚨 HIGH RISK OF FAILURE — Immediate Action Required")
elif df["failure_probability"].iloc[-1] > 0.5:
    st.warning("⚠️ MEDIUM RISK — Monitor Closely")
else:
    st.success("✅ System Healthy")



