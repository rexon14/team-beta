"""
StayScore — Hotel Booking Cancellation Intelligence
Landing page with About, team, and app overview.
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="StayScore — Hotel Cancellation Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("StayScore")
st.markdown("### Hotel Booking Cancellation Intelligence Platform")
st.caption("Predict cancellation risk. Make data-driven decisions.")

st.divider()

# About section
st.header("About")
st.markdown("""
StayScore helps hotel revenue and operations teams anticipate which bookings are likely to be cancelled. 
Built on a Logistic Regression model trained on real hotel booking data, the app predicts cancellation risk 
for individual bookings or bulk samples — so you can prioritize overbooking controls, targeted offers, 
or follow-up actions.
""")

st.subheader("App Features")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Single Prediction**")
    st.caption("Enter booking details via form and get instant prediction with risk percentage.")
with col2:
    st.markdown("**Upload & Predict**")
    st.caption("Upload CSV or Excel with booking samples. Predict in bulk and export results to Excel.")
with col3:
    st.markdown("**Sample Predictions**")
    st.caption("Explore 30 random samples from the dataset with predicted cancellation risk.")

st.divider()

# Team
st.subheader("Created by")
st.markdown("**Group Beta**")
st.markdown("""
- Muhammad Arief Munazat  
- Hanna Muthie Shafira  
- Dennis Schira  
""")

st.caption("Model: Logistic Regression | Dataset: Hotel Bookings")
