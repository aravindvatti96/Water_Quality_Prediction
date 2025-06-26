import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Load and prepare dataset
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df = df.sort_values(by=['id', 'date'])

# Feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear
df['weekofyear'] = df['date'].dt.isocalendar().week
df = df.ffill()

features = ['id', 'NH4', 'BSK5', 'Suspended', 'year', 'month', 'dayofyear', 'weekofyear']
targets = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
df = df.dropna(subset=features + targets)

X = df[features]
y = df[targets]

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit UI
st.set_page_config(page_title="Water Quality Prediction", layout="wide")
st.title("🌊 Water Quality Prediction Dashboard")

# Dataset Preview
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(50))

# Evaluation Metrics
st.subheader("📈 Model Performance")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score (Overall):** {r2:.4f}")

for i, col in enumerate(targets):
    st.write(f"- {col}: R² = {r2_score(y_test[col], y_pred[:, i]):.4f}")

# Feature Importance
st.subheader("🔍 Feature Importance (based on O2)")
importances = model.estimators_[0].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(4, 2))
sns.barplot(x=feat_imp, y=feat_imp.index, ax=ax)
st.pyplot(fig)


# Predict pollutant levels for a specific station and year
station_id = '22'  # change as needed
year_input = 2024  # change as needed

input_data = pd.DataFrame({'year': [year_input], 'id': [station_id]})
input_encoded = pd.get_dummies(input_data, columns=['id'])

# Align with training feature columns
missing_cols = set(X.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[X.columns]  # ensure column order

# Predict pollutants
predicted_pollutants = model.predict(input_encoded)[0]

# Show predicted results
print(f"\nPredicted pollutant levels for station '{station_id}' in {year_input}:")
for p, val in zip(targets, predicted_pollutants):
    st.write(f"  {p}: {val:.2f}")


# Save model and column structure
joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print('\nModel and column structure saved successfully!')


# Custom Prediction Input
st.subheader("🧪 Predict Water Quality From Custom Input")

col1, col2 = st.columns(2)
with col1:
    id_val = st.number_input("Location ID", 1, 22, value=1)
    nh4 = st.number_input("NH4", value=0.5)
    bsk5 = st.number_input("BSK5", value=3.0)
    suspended = st.number_input("Suspended Solids", value=10.0)
with col2:
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2024)
    month = st.slider("Month", 1, 12, 6)

# Calculate derived features
sample_date = datetime(year, month, 15)
dayofyear = sample_date.timetuple().tm_yday
weekofyear = sample_date.isocalendar().week

# Make prediction on input
if st.button("🔮 Predict Water Quality"):
    input_data = pd.DataFrame([[id_val, nh4, bsk5, suspended, year, month, dayofyear, weekofyear]], columns=features)
    prediction = model.predict(input_data)[0]
    result_df = pd.DataFrame([prediction], columns=targets)
    st.write("### 🎯 Prediction Result")
    st.dataframe(result_df.style.format("{:.2f}"))
