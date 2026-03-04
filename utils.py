"""
Shared utilities for StayScore hotel cancellation predictor.
"""

import pickle
import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_resource
def load_model():
    """Load the Logistic Regression model from disk."""
    model_path = Path(__file__).parent / "logreg_tuned.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def row_to_form_data(row):
    """Convert a CSV row (pandas Series) to form_data dict for build_input_df."""
    return {
        "hotel": row["hotel"],
        "lead_time": int(row["lead_time"]),
        "arrival_date_year": int(row["arrival_date_year"]),
        "arrival_date_month": row["arrival_date_month"],
        "arrival_date_week_number": int(row["arrival_date_week_number"]),
        "arrival_date_day_of_month": int(row["arrival_date_day_of_month"]),
        "stays_in_weekend_nights": int(row["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(row["stays_in_week_nights"]),
        "adults": int(row["adults"]),
        "children": int(row["children"]),
        "babies": int(row["babies"]),
        "meal": row["meal"],
        "country": row["country"],
        "market_segment": row["market_segment"],
        "distribution_channel": row["distribution_channel"],
        "is_repeated_guest": "Yes" if row["is_repeated_guest"] == 1 else "No",
        "previous_cancellations": int(row["previous_cancellations"]),
        "previous_bookings_not_canceled": int(row["previous_bookings_not_canceled"]),
        "reserved_room_type": row["reserved_room_type"],
        "assigned_room_type": row["assigned_room_type"],
        "booking_changes": int(row["booking_changes"]),
        "deposit_type": row["deposit_type"],
        "agent": float(row["agent"]),
        "company": float(row["company"]),
        "days_in_waiting_list": int(row["days_in_waiting_list"]),
        "customer_type": row["customer_type"],
        "adr": float(row["adr"]),
        "required_car_parking_spaces": int(row["required_car_parking_spaces"]),
        "total_of_special_requests": int(row["total_of_special_requests"]),
    }


def build_input_df(form_data):
    """Build single-row DataFrame with raw columns for logreg_tuned model input."""
    row = {
        "hotel": form_data["hotel"],
        "lead_time": float(form_data["lead_time"]),
        "arrival_date_year": int(form_data["arrival_date_year"]),
        "arrival_date_month": form_data["arrival_date_month"],
        "arrival_date_week_number": int(form_data["arrival_date_week_number"]),
        "arrival_date_day_of_month": int(form_data["arrival_date_day_of_month"]),
        "stays_in_weekend_nights": int(form_data["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(form_data["stays_in_week_nights"]),
        "adults": int(form_data["adults"]),
        "children": int(form_data["children"]),
        "babies": int(form_data["babies"]),
        "meal": form_data["meal"],
        "country": form_data["country"],
        "market_segment": form_data["market_segment"],
        "distribution_channel": form_data["distribution_channel"],
        "is_repeated_guest": 1 if form_data["is_repeated_guest"] == "Yes" else 0,
        "previous_cancellations": int(form_data["previous_cancellations"]),
        "previous_bookings_not_canceled": int(form_data["previous_bookings_not_canceled"]),
        "reserved_room_type": form_data["reserved_room_type"],
        "assigned_room_type": form_data["assigned_room_type"],
        "booking_changes": int(form_data["booking_changes"]),
        "deposit_type": form_data["deposit_type"],
        "agent": float(form_data["agent"]),
        "company": float(form_data["company"]),
        "days_in_waiting_list": int(form_data["days_in_waiting_list"]),
        "customer_type": form_data["customer_type"],
        "adr": float(form_data["adr"]),
        "required_car_parking_spaces": int(form_data["required_car_parking_spaces"]),
        "total_of_special_requests": int(form_data["total_of_special_requests"]),
    }
    return pd.DataFrame([row])


def build_batch_input_df(df):
    """Build feature DataFrame for batch prediction from raw booking DataFrame."""
    rows = []
    for _, row in df.iterrows():
        form_data = row_to_form_data(row)
        single = build_input_df(form_data)
        rows.append(single)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
