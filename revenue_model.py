import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# PART 1: Engineer the Custom Dataset
# ==========================================
np.random.seed(42) # Ensures we get the same random numbers every time
days = 300 # Simulating about 10 months of daily business data

# Generating random features (Inputs)
ad_spend = np.random.randint(500, 5000, days) # Daily ad spend between ₹500 and ₹5000
discount_pct = np.random.randint(0, 30, days) # Daily discount offered (0% to 30%)
is_weekend = np.random.choice([0, 1], days)   # 0 = Weekday, 1 = Weekend

# Injecting "Real World" Business Logic to calculate Revenue (Output)
# Base revenue + Return on Ad Spend + Discount volume boost + Weekend surge + Random fluctuations
noise = np.random.normal(0, 800, days) # Adding random daily chaos (weather, traffic, etc.)
revenue = 3000 + (ad_spend * 2.5) + (discount_pct * 60) + (is_weekend * 2500) + noise

# Create the DataFrame
df = pd.DataFrame({
    'Ad_Spend_INR': ad_spend,
    'Discount_Pct': discount_pct,
    'Is_Weekend': is_weekend,
    'Daily_Revenue_INR': np.round(revenue, 2) # Rounding to 2 decimal places for currency
})

print("Dataset successfully generated! Here are the first 5 rows:")
print(df.head(), "\n")

# ==========================================
# PART 2: Train the Machine Learning Model
# ==========================================

# Separate Features (X) and Target (y)
X = df[['Ad_Spend_INR', 'Discount_Pct', 'Is_Weekend']]
y = df['Daily_Revenue_INR']

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# PART 3: Evaluate and Extract Business Insights
# ==========================================

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("--- Model Performance ---")
print(f"Accuracy (R-Squared): {r2:.2f} (1.0 is perfect)")
print(f"Average Error: ₹{mae:.2f} per day\n")

print("--- Business Insights (Model Coefficients) ---")
print(f"For every ₹1 spent on Ads, revenue increases by: ₹{model.coef_[0]:.2f}")
print(f"For every 1% increase in Discount, revenue increases by: ₹{model.coef_[1]:.2f}")
print(f"On Weekends, revenue jumps by an average of: ₹{model.coef_[2]:.2f}")
