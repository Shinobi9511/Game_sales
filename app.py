import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

model = pickle.load(open("best_model.pkl", "rb"))
df = pd.read_csv("vgsales.csv")

st.title("🎮 Video Game Global Sales Prediction")
menu = st.sidebar.selectbox(
    "Select Section",
    ["EDA Dashboard", "Prediction", "Feature Importance"]
)
if menu == "EDA Dashboard":
    
    st.header("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    ax.imshow(df.corr(numeric_only=True), cmap='coolwarm')
    st.pyplot(fig)
    
    st.subheader("Sales Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Global_Sales'], bins=50)
    st.pyplot(fig2)
elif menu == "Prediction":
    
    st.header("Enter Game Details")
    
    platform = st.selectbox("Platform", df['Platform'].unique())
    year = st.number_input("Year", 1980, 2025, 2010)
    genre = st.selectbox("Genre", df['Genre'].unique())
    publisher = st.selectbox("Publisher", df['Publisher'].unique())
    na = st.number_input("NA Sales")
    eu = st.number_input("EU Sales")
    jp = st.number_input("JP Sales")
    
    input_data = pd.DataFrame([{
        "Platform": platform,
        "Year": year,
        "Genre": genre,
        "Publisher": publisher,
        "NA_Sales": na,
        "EU_Sales": eu,
        "JP_Sales": jp
    }])
    
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Global Sales: {prediction[0]:.2f} million")
else:
    
    st.header("Feature Importance")
    
    try:
        importances = model.named_steps['model'].feature_importances_
        st.write(importances)
    except:
        st.warning("Feature importance available only for tree-based models.")            