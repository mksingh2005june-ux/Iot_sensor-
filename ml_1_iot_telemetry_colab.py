# IoT Sensor Telemetry Data Cleaning & Normalization
# Streamlit version of the original Google Colab project.

import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="IoT Sensor Telemetry",
    page_icon="📡",
    layout="wide",
)

st.title("📡 IoT Sensor Telemetry Data Cleaning & Normalization")
st.write(
    "Upload IoT temperature and humidity readings to validate, clean, smooth, "
    "normalize, visualize, and download the processed data."
)

# -------------------- Sidebar configuration --------------------
st.sidebar.header("Project configuration")

min_temp = st.sidebar.number_input(
    "Minimum temperature (°C)", value=-40.0, step=1.0
)
max_temp = st.sidebar.number_input(
    "Maximum temperature (°C)", value=85.0, step=1.0
)
min_humidity = st.sidebar.number_input(
    "Minimum humidity (%)", value=0.0, step=1.0
)
max_humidity = st.sidebar.number_input(
    "Maximum humidity (%)", value=100.0, step=1.0
)
apply_smoothing = st.sidebar.checkbox("Apply rolling median smoothing", value=True)
rolling_window = st.sidebar.selectbox(
    "Rolling median window (odd number)",
    options=[3, 5, 7, 9, 11],
    index=0,
    disabled=not apply_smoothing,
)

st.sidebar.caption(
    "These are example sensor limits. Change them to match your actual sensor specifications."
)

# -------------------- Sample dataset --------------------
def make_sample_data():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 09:00:00",
                "2026-01-01 09:01:00",
                "2026-01-01 09:02:00",
                "2026-01-01 09:03:00",
                "2026-01-01 09:04:00",
                "2026-01-01 09:05:00",
                "2026-01-01 09:06:00",
                "2026-01-01 09:07:00",
                "2026-01-01 09:08:00",
                "2026-01-01 09:09:00",
            ],
            "temperature": [
                21.4, 21.6, np.nan, 22.0, 22.2, 22.4, 50.0, 22.8, 23.0, 23.2
            ],
            "humidity": [
                48.2, 48.5, 48.7, 49.1, np.nan, 49.8, 150.0, 50.4, 50.8, 51.1
            ],
        }
    )


# -------------------- Data preparation and processing --------------------
def prepare_telemetry(dataframe):
    required_columns = {"timestamp", "temperature", "humidity"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    data = dataframe[["timestamp", "temperature", "humidity"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
    data["humidity"] = pd.to_numeric(data["humidity"], errors="coerce")

    invalid_timestamp_count = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
