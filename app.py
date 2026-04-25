import streamlit as st
import pickle
import pandas as pd
from src.preprocess import load_data, preprocess_data, split_data

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

st.title("Coal Prediction System")
st.write("Enter details to predict coal production")

# Inputs
state = st.text_input("State/UT Name")

coal_type = st.selectbox("Coal or Lignite", ["Coal", "Lignite"])

ownership = st.selectbox("Ownership", ["Govt Owned", "Private"])

latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)

# Create input dictionary
input_dict = {
    'State/UT Name': state,
    'Coal/Lignite': coal_type,
    'Govt Owned/Private': ownership,
    'Latitude ': latitude,
    'Longitude ': longitude
}

input_df = pd.DataFrame([input_dict])

# Load original data for column alignment
df = load_data()
df = preprocess_data(df)
X, y = split_data(df)

# Convert input same as training
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    # Convert to real-world value
    real_prediction = prediction * 45

    # Categorize
    if real_prediction < 10:
        level = "Low Production"
    elif real_prediction < 25:
        level = "Medium Production"
    else:
        level = "High Production"

    st.success(f"Estimated Coal Production: {round(real_prediction,2)} million tonnes")
    st.info(f"Production Level: {level}")

    st.caption("📌 Note: Prediction is based on historical data patterns and represents estimated coal production in million tonnes.")