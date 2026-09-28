import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. Train the Model Behind the Scenes ---
# (We generate the data and train the model instantly when the app loads)
np.random.seed(42)
days = 300
ad_spend = np.random.randint(500, 5000, days)
discount_pct = np.random.randint(0, 30, days)
is_weekend = np.random.choice([0, 1], days)
noise = np.random.normal(0, 800, days)
revenue = 3000 + (ad_spend * 2.5) + (discount_pct * 60) + (is_weekend * 2500) + noise

df = pd.DataFrame({'Ad_Spend': ad_spend, 'Discount': discount_pct, 'Weekend': is_weekend, 'Revenue': revenue})
X = df[['Ad_Spend', 'Discount', 'Weekend']]
y = df['Revenue']

model = LinearRegression()
model.fit(X, y)

# --- 2. Build the User Interface ---
st.title("🍔 Cloud Kitchen Revenue Predictor")
st.write("Adjust the marketing sliders below to forecast today's expected revenue.")

# Create interactive sliders for the user
input_ad_spend = st.slider("Daily Ad Spend (₹)", min_value=500, max_value=5000, value=2000, step=100)
input_discount = st.slider("Discount Offered (%)", min_value=0, max_value=30, value=10, step=1)
input_weekend = st.radio("Is it a Weekend?", ["No", "Yes"])

# Convert the Yes/No radio button to 1 or 0 for the model
weekend_val = 1 if input_weekend == "Yes" else 0

# --- 3. Make the Prediction ---
if st.button("Predict Revenue"):
    # Format the user's inputs to match our training data
    user_data = pd.DataFrame({
        'Ad_Spend': [input_ad_spend],
        'Discount': [input_discount],
        'Weekend': [weekend_val]
    })
    
    # Predict!
    prediction = model.predict(user_data)[0]
    
    # Display the result beautifully
    st.success(f"📈 Predicted Daily Revenue: ₹{prediction:,.2f}")
    
    st.info(f"**Business Insight:** For every ₹1 spent on ads today, the model expects a return of ₹{model.coef_[0]:.2f}.")