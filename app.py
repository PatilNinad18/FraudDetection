import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

# 1. Collect user input (already done)
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=0.01)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=0.01)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=0.01)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=0.01)

type_options = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
type_choice = st.selectbox("Transaction Type", type_options)

# 2. Build raw input dataframe
input_data = pd.DataFrame([{
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest
}])

# 3. 🔥 Add feature engineering (must match training)
import numpy as np

input_data["errorBalanceOrig"] = input_data["oldbalanceOrg"] - input_data["amount"] - input_data["newbalanceOrig"]
input_data["errorBalanceDest"] = input_data["oldbalanceDest"] + input_data["amount"] - input_data["newbalanceDest"]
input_data["log_amount"] = np.log1p(input_data["amount"])

# If you capped/outlier flagged in training, apply here too
# input_data["amount_capped"] = np.clip(input_data["amount"], lcap, ucap)
# input_data["amount_outlier_flag"] = ((input_data["amount"] < lcap) | (input_data["amount"] > ucap)).astype(int)

# One-hot encode type (same dummy vars as training)
for t in type_options:
    input_data[f"type_{t}"] = 1 if type_choice == t else 0

# 4. Align columns with training
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# 5. Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2f})")
