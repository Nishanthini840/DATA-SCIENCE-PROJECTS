import streamlit as st
import joblib
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

# ---------------- Load Model ----------------
model = joblib.load("random_forest_model.pkl")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main{
    background-color:#f5f7fa;
}
.title{
    font-size:40px;
    font-weight:bold;
    color:#1f77b4;
}
.subtitle{
    font-size:18px;
    color:gray;
}
.stButton>button{
    width:100%;
    background:#1f77b4;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}
.stButton>button:hover{
    background:#0d47a1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<p class="title">🛒 Retail Demand Forecasting Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Future Product Demand using Machine Learning</p>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("📋 Product Information")

category = st.sidebar.number_input("Category",0,20,1)
region = st.sidebar.number_input("Region",0,20,1)
inventory = st.sidebar.number_input("Inventory Level",0,10000,500)
units_sold = st.sidebar.number_input("Units Sold",0,5000,250)
units_ordered = st.sidebar.number_input("Units Ordered",0,5000,300)

price = st.sidebar.number_input("Price",0.0,10000.0,500.0)
discount = st.sidebar.slider("Discount (%)",0.0,100.0,10.0)

weather = st.sidebar.number_input("Weather Condition",0,10,1)
holiday = st.sidebar.number_input("Holiday / Promotion",0,1,0)
competitor = st.sidebar.number_input("Competitor Pricing",0.0,10000.0,450.0)

seasonality = st.sidebar.number_input("Seasonality",0,10,1)
year = st.sidebar.number_input("Year",2023,2035,2024)
month = st.sidebar.slider("Month",1,12,6)
day = st.sidebar.slider("Day",1,31,15)
dayofweek = st.sidebar.slider("Day Of Week",0,6,2)

# ---------------- Dashboard ----------------
col1,col2,col3=st.columns(3)

with col1:
    st.metric("💰 Price",f"₹ {price}")

with col2:
    st.metric("📦 Inventory",inventory)

with col3:
    st.metric("🛍 Units Sold",units_sold)

st.markdown("---")

# ---------------- Prediction ----------------
if st.button("🚀 Predict Demand"):

    data=np.array([[category,
                    region,
                    inventory,
                    units_sold,
                    units_ordered,
                    price,
                    discount,
                    weather,
                    holiday,
                    competitor,
                    seasonality,
                    year,
                    month,
                    day,
                    dayofweek]])

    prediction=model.predict(data)

    st.success("Prediction Completed Successfully ✅")

    st.markdown("## 📈 Predicted Demand")

    st.metric(
        label="Demand Forecast",
        value=f"{prediction[0]:.2f} Units"
    )

    st.balloons()

st.markdown("---")

st.caption("Developed using Streamlit | Random Forest | Machine Learning")