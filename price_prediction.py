import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

os.chdir("D:/practice/Final parali/Final parali/price_prediction")

# Load dataset
try:
    df = pd.read_excel("balanced_demo_data.xlsx")
    print("Data loaded successfully!")
    print("Columns in the dataset:", df.columns)
except Exception as e:
    print("Error loading data:", e)
    exit()

# Encode categorical features
encoder = LabelEncoder()
df["Parali Type"] = encoder.fit_transform(df["Parali Type"])
df["Industry Type"] = encoder.fit_transform(df["Industry Type"])

# Select features and target (fixing column names)
features = ["Parali Type", "Quantity (Tonnes)", "Industry Type", "Buyer Demand (Tonnes)", "Distance (KM)", "Transport Cost"]
target = "Buyer Price (₹/Ton)"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "balanced_price_prediction_model.pkl")
print("Model saved as balanced_price_prediction_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model trained with Mean Absolute Error: {mae:.2f}")

# Predict price based on user input
def predict_price():
    print("\nEnter details to predict stubble price:")
    try:
        parali_type = int(input("Parali Type (0 for Rice, 1 for Wheat): "))
        quantity = float(input("Quantity (Tonnes): "))
        industry_type = int(input("Industry Type (0 for Paper, 1 for Fertilizer, 2 for Biofuel, 3 for Energy): "))
        buyer_demand = float(input("Buyer Demand (Tonnes): "))
        distance = float(input("Distance (KM): "))
        transport_cost = float(input("Transport Cost (₹): "))

        new_data = {
            "Parali Type": parali_type,
            "Quantity (Tonnes)": quantity,
            "Industry Type": industry_type,
            "Buyer Demand (Tonnes)": buyer_demand,
            "Distance (KM)": distance,
            "Transport Cost": transport_cost
        }
        new_data_df = pd.DataFrame([new_data])

        # Load the trained model
        loaded_model = joblib.load("balanced_price_prediction_model.pkl")
        predicted_price = loaded_model.predict(new_data_df)[0]
        print(f"\nPredicted Stubble Price: ₹{predicted_price:.2f} per Ton")
    except Exception as e:
        print("Error during prediction:", e)

if __name__ == "__main__":
    predict_price()

