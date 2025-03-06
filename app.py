import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page config for better layout and appearance
st.set_page_config(page_title="USA Housing Price Predictor", page_icon=":house:")

# Load the dataset
try:
    df = pd.read_csv('USA_Housing.csv')
    df.drop(columns=['Address'], inplace=True)  # Drop Address column
except FileNotFoundError:
    st.error("Error: USA_Housing.csv not found. Please ensure the file is in the same directory.")
    st.stop()


# --- DATA PREPROCESSING ---

# Handle missing values (if any) - simple imputation with the mean
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# --- DATA ANALYSIS & VISUALIZATION ---
st.sidebar.header("Data Exploration & Visualization")

# Display dataset summary statistics
if st.sidebar.checkbox("Show Dataset Summary"):
    st.header("Dataset Summary")
    st.write(df.describe())

# Correlation heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.header("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Pairplot (commented out for performance on large datasets)
# if st.sidebar.checkbox("Show Pairplot"):
#     st.header("Pairplot")
#     fig = sns.pairplot(df)
#     st.pyplot(fig)

# Distribution plots
st.sidebar.subheader("Variable Distributions")
selected_column = st.sidebar.selectbox("Select a column for distribution plot:", df.columns)

if st.sidebar.checkbox(f"Show Distribution of {selected_column}"):
    st.header(f"Distribution of {selected_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], kde=True, ax=ax)
    st.pyplot(fig)

# Scatter plot
st.sidebar.subheader("Scatter Plots")
x_column = st.sidebar.selectbox("Select X axis:", df.columns, index=0)
y_column = st.sidebar.selectbox("Select Y axis:", df.columns, index=5) #preselect Price for Y axis

if st.sidebar.checkbox(f"Show Scatter Plot: {x_column} vs {y_column}"):
    st.header(f"Scatter Plot: {x_column} vs {y_column}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_column, y=y_column, data=df, ax=ax)
    st.pyplot(fig)



# --- MODEL TRAINING ---
st.header("House Price Prediction")
st.write("Enter the features of the house to predict the price.")

# Input fields for features
avg_area_income = st.number_input("Avg. Area Income:", value=68700.0)
avg_area_house_age = st.number_input("Avg. Area House Age:", value=6.0)
avg_area_number_of_rooms = st.number_input("Avg. Area Number of Rooms:", value=7.0)
avg_area_number_of_bedrooms = st.number_input("Avg. Area Number of Bedrooms:", value=4.0)
area_population = st.number_input("Area Population:", value=40000.0)


# Prepare data for model training
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_price(income, house_age, num_rooms, num_bedrooms, population):
    input_data = np.array([[income, house_age, num_rooms, num_bedrooms, population]])
    prediction = model.predict(input_data)
    return prediction[0]


# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(avg_area_income, avg_area_house_age, avg_area_number_of_rooms, avg_area_number_of_bedrooms, area_population)
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")


# --- MODEL EVALUATION ---
st.sidebar.header("Model Evaluation")

if st.sidebar.checkbox("Show Model Evaluation Metrics"):
    st.header("Model Evaluation Metrics")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:,.2f}")
    st.write(f"R-squared: {r2:,.2f}")

# --- ADDITIONAL FEATURES (Optional) ---
st.sidebar.header("About")
st.sidebar.info("This app predicts USA housing prices based on a Linear Regression model. It provides data exploration and visualization tools to understand the dataset.")

# You could also add code to save the model, allow user to upload their own dataset, etc.
