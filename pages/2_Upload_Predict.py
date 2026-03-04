"""
Upload & Predict — Bulk prediction from CSV/Excel upload with table display and Excel export.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from utils import load_model, build_batch_input_df

REQUIRED_COLUMNS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
    "meal", "country", "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "assigned_room_type", "booking_changes", "deposit_type",
    "agent", "company", "days_in_waiting_list", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]

COLUMN_GROUPS = {
    "Hotel & dates": ["hotel", "arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month", "lead_time"],
    "Stay": ["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies"],
    "Booking & customer": ["meal", "country", "market_segment", "distribution_channel", "customer_type", "deposit_type"],
    "Guest history": ["is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled"],
    "Room": ["reserved_room_type", "assigned_room_type", "booking_changes"],
    "Other": ["agent", "company", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests"],
}


def validate_columns(df):
    """Check that required columns exist. Return (ok, missing)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def validate_rows(df):
    """
    Validate rows: drop nulls in required columns, enforce adults >= 1.
    Return (df_valid, df_skipped, list of (row_idx, reason)).
    """
    df = df.copy()
    df["_row_idx"] = range(len(df))

    # Check nulls in required columns
    null_mask = df[REQUIRED_COLUMNS].isna().any(axis=1)
    null_rows = df[null_mask]
    null_reasons = [(int(idx), "Missing required value(s)") for idx in null_rows["_row_idx"]]

    # Adults >= 1
    invalid_adults = df[~null_mask & (df["adults"] < 1)]
    adult_reasons = [(int(idx), "adults must be >= 1") for idx in invalid_adults["_row_idx"]]

    # Valid rows
    valid_mask = ~null_mask & (df["adults"] >= 1)
    df_valid = df[valid_mask][REQUIRED_COLUMNS].copy()
    df_skipped = df[~valid_mask].drop(columns=["_row_idx"], errors="ignore")

    skipped_reasons = null_reasons + adult_reasons
    return df_valid, df_skipped, skipped_reasons


def get_template_df():
    """Load template (1 example row) from hotel_bookings_cleaned.csv."""
    csv_path = Path(__file__).parent.parent / "hotel_bookings_cleaned.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, nrows=1)
    if "is_canceled" in df.columns:
        df = df.drop(columns=["is_canceled"])
    return df[REQUIRED_COLUMNS]


def risk_label(proba):
    pct = proba * 100
    if pct < 30:
        return "Low"
    if pct < 60:
        return "Medium"
    return "High"


st.title("Upload & Predict")
st.caption("Upload CSV or Excel with booking samples. Predict cancellation risk for all rows.")

# Guide expander
with st.expander("Upload guide & requirements", expanded=False):
    st.markdown("""
    **Purpose:** Upload a CSV or Excel file with booking data to predict cancellation risk in bulk. 
    Your file must use the same column structure as our training data.

    **Data format:**
    - Column names must match exactly (case-sensitive).
    - Numeric columns: `lead_time`, `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month`, 
      `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `is_repeated_guest` (0 or 1), 
      `previous_cancellations`, `previous_bookings_not_canceled`, `booking_changes`, `agent`, `company`, 
      `days_in_waiting_list`, `adr`, `required_car_parking_spaces`, `total_of_special_requests`.
    - Categorical columns: `hotel`, `arrival_date_month`, `meal`, `country`, `market_segment`, `distribution_channel`, 
      `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`.
    - Each row must have `adults >= 1` (at least one adult guest).
    - Rows with missing values in required columns are skipped.
    """)

    st.markdown("**Mandatory columns (by category):**")
    for group, cols in COLUMN_GROUPS.items():
        st.markdown(f"- **{group}:** `{', '.join(cols)}`")

    st.markdown("---")
    st.markdown("""
    **Disclaimer:** Predictions are based on a Decision Tree model trained on historical hotel booking data. 
    Results are for informational purposes only and do not guarantee actual cancellation behavior.
    """)

# Template download
template_df = get_template_df()
if template_df is not None:
    col_csv, col_xlsx, _ = st.columns([1, 1, 2])
    with col_csv:
        csv_buf = BytesIO()
        template_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download template (CSV)",
            data=csv_buf.getvalue(),
            file_name="stayscore_upload_template.csv",
            mime="text/csv",
        )
    with col_xlsx:
        xlsx_buf = BytesIO()
        template_df.to_excel(xlsx_buf, index=False, engine="openpyxl")
        xlsx_buf.seek(0)
        st.download_button(
            "Download template (Excel)",
            data=xlsx_buf.getvalue(),
            file_name="stayscore_upload_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.divider()

model = load_model()
if model is None:
    st.error(
        "Model file `decision_tree_model.pkl` not found. "
        "Run the notebook `canceled_bookings_prediction.ipynb` to generate it."
    )
    st.stop()

uploaded = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls"],
    help="Upload CSV or Excel with same columns as hotel_bookings_cleaned.csv (without is_canceled).",
)

if uploaded is None:
    st.info("Upload a CSV or Excel file to get started.")
    st.stop()

# Load file
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.empty:
    st.warning("File is empty.")
    st.stop()

ok, missing = validate_columns(df)
if not ok:
    st.error(f"Missing required columns: {', '.join(missing)}")
    with st.expander("View required columns"):
        for group, cols in COLUMN_GROUPS.items():
            st.markdown(f"**{group}:** `{', '.join(cols)}`")
    st.stop()

# Row-level validation
df_valid, df_skipped, skipped_reasons = validate_rows(df)
n_valid = len(df_valid)
n_skipped = len(df_skipped)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows loaded", len(df))
with col2:
    st.metric("Valid rows", n_valid)
with col3:
    st.metric("Rows skipped", n_skipped)

if n_skipped > 0:
    with st.expander("Show skipped rows", expanded=False):
        st.caption("Rows skipped due to missing values or adults < 1.")
        st.dataframe(df_skipped, use_container_width=True, hide_index=True)

if n_valid == 0:
    st.warning("No valid rows to predict. Ensure each row has adults >= 1 and no missing required values.")
    st.stop()

if st.button("Run Prediction"):
    with st.spinner("Predicting..."):
        try:
            X = build_batch_input_df(df_valid)
            preds = model.predict(X)
            probas = model.predict_proba(X)[:, 1]
        except KeyError as e:
            st.error(f"Column error: {e}. Check that all required columns exist and have valid data.")
            st.stop()
        except ValueError as e:
            st.error(f"Data format error: {e}. Ensure numeric columns contain numbers and categories use expected values.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        result = df_valid.copy()
        result["prediction"] = ["Will Cancel" if p == 1 else "Will Not Cancel" for p in preds]
        result["cancellation_risk_pct"] = [f"{proba * 100:.1f}%" for proba in probas]
        result["risk_label"] = [risk_label(p) for p in probas]

        st.session_state["upload_result"] = result

if "upload_result" in st.session_state:
    result_df = st.session_state["upload_result"]
    st.divider()
    st.subheader("Results")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    buf = BytesIO()
    result_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button(
        "Download as Excel",
        data=buf.getvalue(),
        file_name="stayscore_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
