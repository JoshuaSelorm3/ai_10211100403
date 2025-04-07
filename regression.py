# Name: Joshua Jerry Selorm Yegbe
# Index Number: 10211100403

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def regression_page():
    st.title("Regression Problem")
    st.write("Upload a CSV file and specify the target column for regression.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())

        # Target column selection
        target_col = st.selectbox("Select Target Column", data.columns)
        feature_cols = [col for col in data.columns if col != target_col]

        # Preprocessing: Drop NaN values
        data = data.dropna()
        X = data[feature_cols]
        y = data[target_col]

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Metrics
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Visualization
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color="blue", label="Predictions")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Ideal")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        st.pyplot(fig)

        # Custom Prediction
        st.subheader("Make a Prediction")
        custom_input = {}
        for col in feature_cols:
            custom_input[col] = st.number_input(f"Enter {col}", value=0.0)
        if st.button("Predict"):
            input_df = pd.DataFrame([custom_input])
            pred = model.predict(input_df)[0]
            st.write(f"Predicted {target_col}: {pred:.2f}")

if __name__ == "__main__":
    regression_page()