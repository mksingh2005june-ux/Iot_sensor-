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

    for column, (lower_limit, upper_limit) in sensor_bounds.items():
        values = pd.to_numeric(cleaned[column], errors="coerce")
        outside_limits = values.notna() & (
            (values < lower_limit) | (values > upper_limit)
        )

        report["invalid_values_by_column"][column] = int(outside_limits.sum())
        values = values.mask(outside_limits)

        report["missing_before_interpolation"][column] = int(values.isna().sum())
        values = values.interpolate(method="linear", limit_direction="both")

        if values.isna().all():
            raise ValueError(
                f"Column '{column}' has no valid readings to interpolate."
            )

        cleaned[column] = values
        report["missing_after_interpolation"][column] = int(values.isna().sum())

    if rolling_median_window is not None:
        if rolling_median_window < 3 or rolling_median_window % 2 == 0:
            raise ValueError("Rolling median window must be an odd integer >= 3.")

        for column in sensor_bounds:
            cleaned[column] = cleaned[column].rolling(
                window=rolling_median_window,
                center=True,
                min_periods=1
            ).median()

    report["rows_after_cleaning"] = int(len(cleaned))
    return cleaned, report


def minmax_normalize(dataframe, columns=("temperature", "humidity")):
    normalized = dataframe.copy()
    scaling_parameters = {}

    for column in columns:
        values = normalized[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} still contains missing or non-finite values."
            )

        minimum = float(np.min(values))
        maximum = float(np.max(values))
        value_range = maximum - minimum

        if np.isclose(value_range, 0.0):
            scaled_values = np.zeros_like(values, dtype=float)
        else:
            scaled_values = (values - minimum) / value_range

        normalized[f"{column}_scaled"] = scaled_values
        scaling_parameters[column] = {"min": minimum, "max": maximum}

    return normalized, scaling_parameters


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def json_to_bytes(data):
    return json.dumps(data, indent=4).encode("utf-8")


if temp_min >= temp_max or humidity_min >= humidity_max:
    st.warning("Correct the sensor limits in the sidebar before processing.")
elif uploaded_file is not None and not use_sample:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
elif use_sample:
    raw_df = make_sample_data()
else:
    st.info("Upload a CSV file or select 'Use built-in sample dataset' to begin.")
    st.stop()

st.subheader("2. Raw sensor data")
st.dataframe(raw_df, use_container_width=True)
st.caption(f"Rows: {len(raw_df)}")

if st.button("Run data cleaning and normalization", type="primary"):
    try:
        prepared_df, invalid_timestamp_count = prepare_telemetry(raw_df)

        cleaned_df, cleaning_report = clean_telemetry(
            prepared_df,
            SENSOR_BOUNDS,
            invalid_timestamps_dropped=invalid_timestamp_count,
            rolling_median_window=ROLLING_MEDIAN_WINDOW
        )

        final_df, scaling_parameters = minmax_normalize(cleaned_df)

        assert final_df[["temperature", "humidity"]].isna().sum().sum() == 0
        assert final_df["temperature_scaled"].between(0, 1).all()
        assert final_df["humidity_scaled"].between(0, 1).all()

        processing_report = {
            **cleaning_report,
            "scaling_parameters": scaling_parameters,
            "output_rows": int(len(final_df)),
            "output_columns": list(final_df.columns)
        }

        st.success("Pipeline completed successfully. Validation passed.")

        st.subheader("3. Cleaned and normalized data")
        st.dataframe(final_df, use_container_width=True)

        st.subheader("4. Sensor readings")
        chart_data = final_df.set_index("timestamp")[["temperature", "humidity"]]
        st.line_chart(chart_data)

        st.subheader("5. Normalized feature ranges")
        ranges = final_df[
            ["temperature_scaled", "humidity_scaled"]
        ].agg(["min", "max"])
        st.dataframe(ranges, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows received", len(raw_df))
        col2.metric("Rows after cleaning", len(final_df))
        col3.metric("Invalid timestamps dropped", invalid_timestamp_count)

        st.subheader("6. Processing report")
        st.json(processing_report)

        st.subheader("7. Download results")
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "⬇️ Download cleaned CSV",
                data=dataframe_to_csv_bytes(final_df),
                file_name="cleaned_telemetry.csv",
                mime="text/csv"
            )
        with download_col2:
            st.download_button(
                "⬇️ Download processing report (JSON)",
                data=json_to_bytes(processing_report),
                file_name="processing_report.json",
                mime="application/json"
            )

    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.exception(exc)
