import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page configuration
st.set_page_config(page_title="House Price Prediction", layout="wide")

# ---------------- APP TITLE ----------------
st.title("🏠 House Price Prediction System")
st.write("AI powered system to estimate house prices based on property features.")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload House Dataset (CSV)", type="csv")

if uploaded_file is not None:

    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # ---------------- DATASET DASHBOARD ----------------
    st.subheader("Dataset Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Houses", data.shape[0])
    col2.metric("Total Features", data.shape[1] - 1)
    col3.metric("Average Price", round(data.iloc[:, -1].mean(), 2))

    # ---------------- CORRELATION ----------------
    st.subheader("Feature Correlation")

    corr = data.corr(numeric_only=True)
    st.dataframe(corr)

    # Features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert categorical values
    X = pd.get_dummies(X)

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # ---------------- MODEL PERFORMANCE ----------------
    st.subheader("Model Performance")

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    m1, m2, m3 = st.columns(3)

    m1.metric("R² Score", round(r2, 3))
    m2.metric("MAE", round(mae, 2))
    m3.metric("RMSE", round(rmse, 2))

    # ---------------- LINE GRAPH ----------------
    st.subheader("Actual vs Predicted Prices")

    graph_df = pd.DataFrame({
        "Actual Price": y.values,
        "Predicted Price": y_pred
    })

    graph_df = graph_df.sort_values(by="Actual Price")

    st.line_chart(graph_df)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Impact": model.coef_
    })

    importance = importance.sort_values("Impact", ascending=False)

    st.bar_chart(importance.set_index("Feature"))

    # ---------------- DATA FILTER ----------------
    st.subheader("Explore Dataset")

    price_column = data.columns[-1]

    min_price = int(data[price_column].min())
    max_price = int(data[price_column].max())

    selected_price = st.slider(
        "Filter houses by price",
        min_price,
        max_price,
        (min_price, max_price)
    )

    filtered_data = data[
        (data[price_column] >= selected_price[0]) &
        (data[price_column] <= selected_price[1])
    ]

    st.dataframe(filtered_data)

    # ---------------- INPUT FEATURES ----------------
    st.subheader("Enter House Features")

    inputs = []

    for col in X.columns:
        val = st.number_input(col, value=float(X[col].mean()))
        inputs.append(val)

    # ---------------- PREDICTION ----------------
    if st.button("Predict House Price"):

        prediction = model.predict([inputs])[0]

        price_lakh = prediction / 100000

        st.success(f"Estimated Price: ₹ {round(price_lakh, 2)} Lakhs")

    # ---------------- DOWNLOAD RESULTS ----------------
    st.subheader("Download Prediction Results")

    result_df = pd.DataFrame({
        "Actual Price": y,
        "Predicted Price": y_pred
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="house_price_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a dataset to start the prediction system.")