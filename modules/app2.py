# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from modules.data_loader import load_processed_data
from modules.predictor import predict_emission
from modules.visualizer import plot_feature_importance
from config import ALL_FEATURES, BEST_MODEL_PATH

st.set_page_config(page_title="CO2 Emission Predictor Dashboard", layout="wide")
st.title("📊 CO₂ Emission Analysis & Prediction for Telecom Base Stations")

# Load data and model
try:
    data = load_processed_data()
    model = joblib.load(BEST_MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Tabs for dashboard layout
tabs = st.tabs(["📈 Data Overview", "📉 Data Visualization", "🤖 Prediction"])

# --- TAB 1: Data Overview ---
with tabs[0]:
    st.subheader("Raw Data Sample")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    st.markdown("---")
    st.subheader("Summary Statistics")
    st.write(data.describe(include='all'))

# --- TAB 2: Data Visualization ---
with tabs[1]:
    st.subheader("Feature Distributions")
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Select a numerical feature to visualize", numeric_cols)
    fig = px.histogram(data, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CO2 Emission vs. Selected Feature")
    feature = st.selectbox("Select feature for comparison", ALL_FEATURES)
    fig2 = px.scatter(data, x=feature, y="co2_emission", trendline="ols",
                      title=f"CO2 Emission vs {feature.title()}")
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: Prediction ---
with tabs[2]:
    st.subheader("Input Parameters")
    user_input = {}

    for feature in ALL_FEATURES:
        if data[feature].dtype == 'object':
            user_input[feature] = st.selectbox(
                f"{feature.title()}", sorted(data[feature].dropna().unique())
            )
        else:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            mean_val = float(data[feature].mean())
            user_input[feature] = st.slider(
                f"{feature.title()}", min_val, max_val, mean_val
            )

    input_df = pd.DataFrame([user_input])

    if st.button("🚀 Predict CO₂ Emission"):
        prediction = predict_emission(model, input_df)
        st.success(f"Estimated CO₂ Emission: **{prediction[0]:.2f} kg**")

    st.subheader("Feature Importance")
    fig = plot_feature_importance(model, data[ALL_FEATURES])
    st.plotly_chart(fig, use_container_width=True)