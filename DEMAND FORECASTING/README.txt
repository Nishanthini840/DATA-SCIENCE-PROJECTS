# 🛒 Retail Demand Forecasting

## 📌 Project Overview
Retail Demand Forecasting is a Machine Learning project that predicts future product demand based on historical retail data. This helps retailers optimize inventory, reduce stock shortages, and improve business decision-making.

## 🚀 Features
- Predict future product demand
- Interactive Streamlit web application
- Random Forest Machine Learning model
- User-friendly interface
- Real-time demand prediction

## 🛠 Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## 📂 Project Structure

```
Retail-Demand-Forecasting/
│── app.py
│── random_forest_model.pkl
│── retail_store_inventory.csv
│── requirements.txt
│── README.md
```

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Retail-Demand-Forecasting.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📊 Input Features

- Category
- Region
- Inventory Level
- Units Sold
- Units Ordered
- Price
- Discount
- Weather Condition
- Holiday / Promotion
- Competitor Pricing
- Seasonality
- Year
- Month
- Day
- Day of Week

## 🎯 Output

The model predicts the expected product demand (Units).