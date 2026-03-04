"""
Sample Predictions — 30 random samples with predicted cancellation risk and Excel export.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from utils import load_model, build_batch_input_df


def load_samples(n=30):
    """Load n random samples from CSV."""
    csv_path = Path(__file__).parent.parent / "hotel_bookings_cleaned.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df_valid = df[df["adults"] >= 1]
    if df_valid.empty:
        df_valid = df
    sample_df = df_valid.sample(n=min(n, len(df_valid)), replace=False)
    return sample_df


def risk_label(proba):
    pct = proba * 100
    if pct < 30:
        return "Low"
    if pct < 60:
        return "Medium"
    return "High"


st.title("Sample Predictions")
st.caption("30 random booking samples with predicted cancellation risk.")

model = load_model()
if model is None:
    st.error(
        "Model file `decision_tree_model.pkl` not found. "
        "Run the notebook `canceled_bookings_prediction.ipynb` to generate it."
    )
    st.stop()

if st.button("Refresh Samples"):
    st.rerun()

sample_df = load_samples(30)
if sample_df is None:
    st.error("Could not load hotel_bookings_cleaned.csv")
    st.stop()

# Backend: build features and predict
X = build_batch_input_df(sample_df)
preds = model.predict(X)
probas = model.predict_proba(X)[:, 1]

# Full result: all original columns + prediction columns
result_df = sample_df.copy()
result_df["cancellation_risk_pct"] = [f"{p * 100:.1f}%" for p in probas]
result_df["prediction"] = ["Will Cancel" if p == 1 else "Will Not Cancel" for p in preds]
result_df["risk_label"] = [risk_label(p) for p in probas]

st.dataframe(result_df, use_container_width=True, hide_index=True)

buf = BytesIO()
result_df.to_excel(buf, index=False, engine="openpyxl")
buf.seek(0)
st.download_button(
    "Download as Excel",
    data=buf.getvalue(),
    file_name="stayscore_sample_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
