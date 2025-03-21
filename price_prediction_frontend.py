import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load("balanced_price_prediction_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Predict price
def predict_price(new_data):
    new_data_df = pd.DataFrame([new_data])
    predicted_price = model.predict(new_data_df)[0]
    return predicted_price

# Streamlit page configuration
st.set_page_config(
    page_title="Parali Price Prediction",
    page_icon="🌾",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #888;
        }
        .predict-btn button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .result {
            font-size: 28px;
            font-weight: bold;
            color: #0E76A8;
            text-align: center;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #aaa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown('<div class="title">🌾 Parali Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find the optimal selling price for your stubble waste</div>', unsafe_allow_html=True)

# User inputs with streamlined UI
st.markdown("### 🌟 Enter Stubble Details")
col1, col2 = st.columns(2)

with col1:
    parali_type = st.selectbox("Parali Type", ["Rice", "Wheat"], help="Select the type of stubble")
    industry_type = st.selectbox("Industry Type", ["Paper", "Fertilizer", "Biofuel", "Energy"], help="Select the industry")
    distance = st.number_input("Distance (KM)", min_value=0.0, max_value=1000.0, value=50.0, step=1.0, help="Distance from source to destination")

with col2:
    quantity = st.number_input("Quantity (Tonnes)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0, help="Total stubble quantity")
    buyer_demand = st.number_input("Buyer Demand (Tonnes)", min_value=1.0, max_value=10000.0, value=500.0, step=1.0, help="Market demand for stubble")
    transport_cost = st.number_input("Transport Cost", min_value=0.0, max_value=10000.0, value=1000.0, step=100.0, help="Transportation cost")

# Predict button with animation
if st.button("Predict Price", key="predict-btn"):
    # Encode the categorical inputs
    parali_type_encoded = 0 if parali_type == "Rice" else 1
    industry_type_encoded = ["Paper", "Fertilizer", "Biofuel", "Energy"].index(industry_type)

    # Prepare the input dictionary
    new_data = {
        "Parali Type": parali_type_encoded,
        "Quantity (Tonnes)": quantity,
        "Industry Type": industry_type_encoded,
        "Buyer Demand (Tonnes)": buyer_demand,
        "Distance (KM)": distance,
        "Transport Cost": transport_cost
    }

    # Predict and display the result
    try:
        predicted_price = predict_price(new_data)
        st.markdown(f'<div class="result">💰 Predicted Stubble Price: ₹{predicted_price:.2f} per Ton</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
