import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("neo_classifier.pkl")

# --- Page Config ---
st.set_page_config(page_title="NEO Hazard Classifier", page_icon="☄️", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        body {
            background-color: #f6f8fa;
            color: #333;
        }
        .main {
            padding: 2rem;
        }
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #1f4e79;
        }
        .stButton>button {
            background-color: #1f4e79;
            color: white;
            font-weight: 600;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("☄️ NEO Hazard Classifier")
st.caption("Predict whether a Near-Earth Object is potentially hazardous based on key features.")

# --- Input Form ---
with st.form("neo_form"):
    st.subheader("Enter NEO Parameters")

    diameter = st.number_input("Estimated Diameter (km)", min_value=0.0, value=1.0, format="%.2f")
    velocity = st.number_input("Relative Velocity (km/s)", min_value=0.0, value=2.0, format="%.2f")
    miss_distance = st.number_input("Miss Distance (km)", min_value=0.0, value=3.0, format="%.2f")
    magnitude = st.number_input("Absolute Magnitude", min_value=0.0, value=22.0, format="%.2f")

    submit = st.form_submit_button("Predict 🚀")

# --- Transform Inputs ---
# Apply log1p to match training preprocessing

transformed_inputs = np.log1p ([
    diameter,
    velocity,
    miss_distance
])

# --- Prediction Logic ---
if submit:
    features = np.array([[transformed_inputs, magnitude]])
    prediction = model.predict(features)[0]

    st.subheader("🛰️ Prediction Result")
    if prediction == 1:
        st.error("⚠️ Potentially Hazardous!")
    else:
        st.success("✅ Not Hazardous")

# --- Footer ---
# Footer
st.markdown("---")
st.markdown("Built with ❤️ using `scikit-learn` + `Streamlit` | [View on GitHub](https://github.com/katekolomii/neo-asteroid-classifier)")
