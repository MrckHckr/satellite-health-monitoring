import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from lifelines import KaplanMeierFitter

# Caricamento risorse
st.set_page_config(page_title="Satellite Health Monitoring", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/TUO_USERNAME/satellite-health-monitoring/main/data/raw/telemetry.csv"
    return pd.read_csv(url, parse_dates=["timestamp"])

@st.cache_resource
def load_model_assets():
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

# Header
st.title("🛰️ Satellite Health Monitoring Dashboard")
st.subheader("Real-Time Predictive Maintenance & Decision Support")

# Telemetria live
df = load_data()

st.metric("Samples", len(df))
st.metric("Subsystems", df["subsystem"].nunique())

st.line_chart(
    df.set_index("timestamp")[["temperature", "vibration", "power"]]
)

# Anomaly Detection (Isolation Forest)
from sklearn.ensemble import IsolationForest

features = df[["temperature", "vibration", "power"]]

iso = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = iso.fit_predict(features)

anomalies = df[df["anomaly"] == -1]

st.subheader("🚨 Anomaly Detection")
st.write(f"Detected anomalies: {len(anomalies)}")
st.dataframe(anomalies.tail())

# Failure Probability (LSTM)
model, scaler = load_model_assets()

WINDOW_SIZE = 3

def create_sequence(data):
    return np.expand_dims(data[-WINDOW_SIZE:], axis=0)

scaled = scaler.transform(features)
X_inference = create_sequence(scaled)

failure_prob = model.predict(X_inference, verbose=0)[0][0]

st.subheader("🔮 Failure Prediction")
st.metric("Failure Probability (next horizon)", f"{failure_prob:.2f}")

# Decision Support
if failure_prob > 0.6:
    st.error("🚨 ACTION REQUIRED: Schedule maintenance")
elif failure_prob > 0.4:
    st.warning("⚠️ Increase monitoring")
else:
    st.success("✅ System nominal")

# Survival Analysis (RUL)
st.subheader("⏳ Remaining Useful Life")

df_sorted = df.sort_values("timestamp")
df_sorted["time"] = (df_sorted["timestamp"] - df_sorted["timestamp"].min()).dt.total_seconds() / 3600

kmf = KaplanMeierFitter()
kmf.fit(df_sorted["time"], event_observed=df_sorted["failed"])

fig, ax = plt.subplots()
kmf.plot_survival_function(ax=ax)
ax.set_xlabel("Operating Time (hours)")
ax.set_ylabel("Survival Probability")

st.pyplot(fig)

# Avvio locale
streamlit run app/dashboard.py
