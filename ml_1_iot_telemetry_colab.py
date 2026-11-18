import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="IoT Sensor Telemetry Cleaner",
    page_icon="📡",
    layout="wide"
)

st.title("📡 IoT Sensor Telemetry Data Cleaning & Normalization")
st.write(
    "Upload IoT temperature and humidity readings to validate, clean, "
    "smooth, normalize, and download the processed data."
)

st.sidebar.header("Project configuration")
temp_min = st.sidebar.number_input("Minimum temperature (°C)", value=-40.0)
temp_max = st.sidebar.number_input("Maximum temperature (°C)", value=85.0)
humidity_min = st.sidebar.number_input("Minimum humidity (%)", value=0.0)
humidity_max = st.sidebar.number_input("Maximum humidity (%)", value=100.0)

if temp_min >= temp_max:
    st.sidebar.error("Maximum temperature must be greater than minimum.")
if humidity_min >= humidity_max:
    st.sidebar.error("Maximum humidity must be greater than minimum.")

use_smoothing = st.sidebar.checkbox("Apply rolling median smoothing", value=True)
rolling_window = 3
if use_smoothing:
    rolling_window = st.sidebar.selectbox(
        "Rolling median window (odd number)",
        options=[3, 5, 7, 9],
        index=0
    )

SENSOR_BOUNDS = {
    "temperature": (temp_min, temp_max),
    "humidity": (humidity_min, humidity_max)
}
ROLLING_MEDIAN_WINDOW = rolling_window if use_smoothing else None

st.subheader("1. Upload your sensor dataset")
st.caption(
    "CSV must contain these columns: timestamp, temperature, humidity. "
    "If you do not upload a file, you can run the included sample dataset."
)

uploaded_file = st.file_uploader("Upload IoT sensor CSV", type=["csv"])
use_sample = st.checkbox("Use built-in sample dataset", value=uploaded_file is None)

def make_sample_data():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:00:00", periods=10, freq="min"),
        "temperature": [21.4, 21.6, np.nan, 22.0, 22.2, 22.4, 50.0, 22.8, 23.0, 23.2],
        "humidity": [48.2, 48.5, 48.7, 49.1, np.nan, 49.8, 150.0, 50.4, 50.8, 51.1]
    })


def prepare_telemetry(dataframe):
    required_columns = {"timestamp", "temperature", "humidity"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = dataframe.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
    data["humidity"] = pd.to_numeric(data["humidity"], errors="coerce")

    invalid_timestamp_count = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        raise ValueError("No rows with valid timestamps were found.")

    return data, invalid_timestamp_count


def clean_telemetry(dataframe, sensor_bounds, invalid_timestamps_dropped=0,
                    rolling_median_window=None):
    cleaned = dataframe.copy()
    report = {
        "rows_received": int(len(cleaned)),
        "invalid_timestamps_dropped": int(invalid_timestamps_dropped),
        "invalid_values_by_column": {},
        "missing_before_interpolation": {},
        "missing_after_interpolation": {},
        "rolling_median_window": rolling_median_window
    }
