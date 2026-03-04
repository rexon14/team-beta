"""
Single Prediction — Form-based cancellation prediction with Excel export.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from utils import load_model, build_input_df, row_to_form_data

# Load categorical options from CSV
@st.cache_data
def load_categorical_options():
    csv_path = Path(__file__).parent.parent / "hotel_bookings_cleaned.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    return {
        "hotel": sorted(df["hotel"].dropna().unique().tolist()),
        "arrival_date_month": [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ],
        "meal": sorted(df["meal"].dropna().unique().tolist()),
        "country": sorted(df["country"].dropna().unique().tolist()),
        "market_segment": sorted(df["market_segment"].dropna().unique().tolist()),
        "distribution_channel": sorted(df["distribution_channel"].dropna().unique().tolist()),
        "reserved_room_type": sorted(df["reserved_room_type"].dropna().unique().tolist()),
        "assigned_room_type": sorted(df["assigned_room_type"].dropna().unique().tolist()),
        "deposit_type": sorted(df["deposit_type"].dropna().unique().tolist()),
        "customer_type": sorted(df["customer_type"].dropna().unique().tolist()),
    }


def get_random_sample(cat_options):
    """Sample a random row from the dataset for form defaults."""
    csv_path = Path(__file__).parent.parent / "hotel_bookings_cleaned.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df_valid = df[df["adults"] >= 1]
    if df_valid.empty:
        df_valid = df
    row = df_valid.sample(n=1).iloc[0]
    return row_to_form_data(row)


def risk_label(proba):
    """Return risk label from probability."""
    pct = proba * 100
    if pct < 30:
        return "Low"
    if pct < 60:
        return "Medium"
    return "High"


st.title("Single Prediction")
st.caption("Enter booking details and predict cancellation risk.")

model = load_model()
if model is None:
    st.error(
        "Model file `decision_tree_model.pkl` not found. "
        "Run the notebook `canceled_bookings_prediction.ipynb` to generate it."
    )
    st.stop()

cat_options = load_categorical_options()
if not cat_options:
    st.warning("Could not load categorical options from hotel_bookings_cleaned.csv")
    st.stop()

if st.button("Generate Random Values"):
    sample = get_random_sample(cat_options)
    if sample:
        st.session_state["form_defaults"] = sample
        st.rerun()

defaults = st.session_state.get("form_defaults", {})

with st.form("prediction_form"):
    col1, col2, col3, col4, col5 = st.columns(5)

    def _idx(opts, val):
        return opts.index(val) if val in opts else 0

    with col1:
        hotel = st.selectbox(
            "Hotel", cat_options["hotel"],
            index=_idx(cat_options["hotel"], defaults.get("hotel", cat_options["hotel"][0]))
        )
        lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=730, value=defaults.get("lead_time", 30))
        arrival_date_year = st.number_input("Arrival Year", min_value=2015, max_value=2017, value=defaults.get("arrival_date_year", 2016))
        arrival_date_month = st.selectbox(
            "Arrival Month", cat_options["arrival_date_month"],
            index=_idx(cat_options["arrival_date_month"], defaults.get("arrival_date_month", "July"))
        )
        arrival_date_week_number = st.number_input("Arrival Week Number", min_value=1, max_value=53, value=defaults.get("arrival_date_week_number", 27))
        arrival_date_day_of_month = st.number_input("Arrival Day of Month", min_value=1, max_value=31, value=defaults.get("arrival_date_day_of_month", 15))

    with col2:
        stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=7, value=defaults.get("stays_in_weekend_nights", 0))
        stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=14, value=defaults.get("stays_in_week_nights", 2))
        adults = st.number_input("Adults", min_value=1, max_value=4, value=defaults.get("adults", 2))
        children = st.number_input("Children", min_value=0, max_value=3, value=defaults.get("children", 0))
        babies = st.number_input("Babies", min_value=0, max_value=2, value=defaults.get("babies", 0))
        meal = st.selectbox("Meal", cat_options["meal"], index=_idx(cat_options["meal"], defaults.get("meal", "BB")))

    with col3:
        country = st.selectbox("Country", cat_options["country"], index=_idx(cat_options["country"], defaults.get("country", "PRT")))
        market_segment = st.selectbox(
            "Market Segment", cat_options["market_segment"],
            index=_idx(cat_options["market_segment"], defaults.get("market_segment", "Direct"))
        )
        distribution_channel = st.selectbox(
            "Distribution Channel", cat_options["distribution_channel"],
            index=_idx(cat_options["distribution_channel"], defaults.get("distribution_channel", "Direct"))
        )
        is_repeated_guest = st.selectbox("Repeated Guest", ["No", "Yes"], index=1 if defaults.get("is_repeated_guest") == "Yes" else 0)
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=defaults.get("previous_cancellations", 0))
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Canceled", min_value=0, max_value=50, value=defaults.get("previous_bookings_not_canceled", 0)
        )

    with col4:
        reserved_room_type = st.selectbox(
            "Reserved Room Type", cat_options["reserved_room_type"],
            index=_idx(cat_options["reserved_room_type"], defaults.get("reserved_room_type", "C"))
        )
        assigned_room_type = st.selectbox(
            "Assigned Room Type", cat_options["assigned_room_type"],
            index=_idx(cat_options["assigned_room_type"], defaults.get("assigned_room_type", "C"))
        )
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=10, value=defaults.get("booking_changes", 0))
        deposit_type = st.selectbox(
            "Deposit Type", cat_options["deposit_type"],
            index=_idx(cat_options["deposit_type"], defaults.get("deposit_type", "No Deposit"))
        )
        agent = st.number_input("Agent ID", min_value=0.0, max_value=500.0, value=defaults.get("agent", 0.0), step=1.0)
        company = st.number_input("Company ID", min_value=0.0, max_value=500.0, value=defaults.get("company", 0.0), step=1.0)

    with col5:
        days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=400, value=defaults.get("days_in_waiting_list", 0))
        customer_type = st.selectbox(
            "Customer Type", cat_options["customer_type"],
            index=_idx(cat_options["customer_type"], defaults.get("customer_type", "Transient"))
        )
        adr = st.number_input("ADR (Avg Daily Rate)", min_value=0.0, max_value=500.0, value=defaults.get("adr", 75.0), step=5.0)
        required_car_parking_spaces = st.number_input("Car Parking Spaces", min_value=0, max_value=3, value=defaults.get("required_car_parking_spaces", 0))
        total_of_special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=defaults.get("total_of_special_requests", 0))

    submitted = st.form_submit_button("Predict")

if submitted:
    form_data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
    }

    try:
        X = build_input_df(form_data)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.error("**Will Cancel**")
    else:
        st.success("**Will Not Cancel**")

    risk_pct = proba * 100
    label = risk_label(proba)
    st.metric("Cancellation Risk", f"{risk_pct:.1f}%", delta=label)

    # Export to Excel
    export_df = pd.DataFrame([{
        **form_data,
        "prediction": "Will Cancel" if pred == 1 else "Will Not Cancel",
        "cancellation_risk_pct": f"{risk_pct:.1f}%",
        "risk_label": label,
    }])
    buf = BytesIO()
    export_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button(
        "Download as Excel",
        data=buf.getvalue(),
        file_name="stayscore_single_prediction.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
