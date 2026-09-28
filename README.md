# 📈 Cloud Kitchen Revenue Predictor 

## 📌 Project Overview
A lightweight, end-to-end Machine Learning project built to solve a real-world micro-business problem. This project uses **Linear Regression** to forecast daily food delivery revenue for a local cloud kitchen based on their marketing spend and discount strategies.

Unlike standard portfolio projects that rely on pre-cleaned Kaggle datasets, I engineered a **custom synthetic dataset** using Python and Pandas to simulate realistic daily operations, including ad-spend, discount rates, weekend surges, and random environmental noise.

## 🛠 Tech Stack
* **Language:** Python
* **Data Engineering:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Metrics:** Mean Absolute Error (MAE), R-Squared

## 💡 The Business Problem
A local cloud kitchen runs ads on delivery platforms but lacks visibility into how their daily ad spend and discount percentages actually translate into final daily revenue. 

## ⚙️ The Solution & Approach
1. **Data Generation:** Simulated 300 days of business operations.
2. **Model Training:** Deployed a Scikit-Learn Linear Regression model to find the mathematical relationship between the marketing inputs and the revenue output.
3. **Business Insights Extracted:** By analyzing the model's coefficients, the system acts as a Return on Ad Spend (ROAS) calculator, proving the exact monetary value of every ₹1 spent on platform ads.

## 🚀 How to Run
The entire pipeline (data generation, model training, and evaluation) is contained within `revenue_model.py`.
