import streamlit as st
import numpy as np
import joblib


model = joblib.load("rf_tuned.pkl")


st.set_page_config(
    page_title="MediSure AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 MediSure AI")
st.subheader("Health Insurance Cost Predictor")

st.write(
    "Enter customer details below to estimate the expected insurance cost."
)


age = st.number_input("Age", min_value=18, max_value=100, value=21)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=22.5
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["Northwest", "Northeast", "Southwest", "Southeast"]
)


sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

region_map = {
    "Northwest": 0,
    "Northeast": 1,
    "Southwest": 2,
    "Southeast": 3
}

region = region_map[region]

if st.button("Predict Insurance Cost"):

    features = np.array(
        [[
            age,
            sex,
            bmi,
            children,
            smoker,
            region
        ]]
    )

    pred = model.predict(features)[0]

    if pred < 0:
        st.error("Error calculating Amount!")
    else:
        st.success(
            f"Expected Insurance Amount: ₹ {pred:,.2f}"
        )

st.markdown("---")
st.caption(
    "Medical Insurance Cost Prediction System | Developed by Hriday Sharma"
)

