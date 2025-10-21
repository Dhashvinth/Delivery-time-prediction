import pandas as pd
import numpy as np
import pickle
import myModule as mm
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\DELL\Downloads\amazon_delivery (1).csv")


y = df["Delivery_Time"]
X = df.drop(columns=["Delivery_Time"])

# Drop identifier-like columns
id_cols = [c for c in X.columns if any(k in c.lower() for k in ['id','order_id','tracking','awb','invoice'])]
X.drop(columns=id_cols, inplace=True, errors='ignore')


numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()
numeric_features = [c for c in numeric_features if not c.endswith('_dt')]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -----------------------------------------------------------------
# STEP 5: TRAIN/TEST SPLIT
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📚 Split -> Train: {X_train.shape}, Test: {X_test.shape}")

# -----------------------------------------------------------------
# STEP 6: MODEL TRAINING
# -----------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=20, n_jobs=-1, random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    r2 = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'R2': r2}
    print(f"{name}: MAE={mae:.3f},  R2={r2:.3f}")

# -----------------------------------------------------------------
# STEP 7: SAVE BEST MODEL
# -----------------------------------------------------------------
best_model_name = min(results, key=lambda m: results[m]['MAE'])
best_model = models[best_model_name]

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])
final_pipe.fit(X_train, y_train)

with open(MODEL_FILE, "wb") as f:
    pickle.dump({'pipeline': final_pipe, 'target_col': TARGET}, f)


import streamlit as st
import pandas as pd
import pickle

MODEL_FILE = "{MODEL_FILE}"

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj["target_col"]

pipe, target = load_model()
st.title("📦 E-commerce Delivery Time Predictor")

st.sidebar.header("🧾 Enter Order Details")
agent_age = st.sidebar.number_input("Agent Age", value=25)
agent_rating = st.sidebar.number_input("Agent Rating", value=4.5)
store_lat = st.sidebar.number_input("Store Latitude", value=12.97)
store_lon = st.sidebar.number_input("Store Longitude", value=77.59)
drop_lat = st.sidebar.number_input("Drop Latitude", value=12.90)
drop_lon = st.sidebar.number_input("Drop Longitude", value=77.60)
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Foggy"])
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High"])
vehicle = st.sidebar.selectbox("Vehicle", ["Bike", "Car", "Van"])
area = st.sidebar.selectbox("Area", ["Urban", "Suburban", "Rural"])
category = st.sidebar.selectbox("Category", ["Grocery", "Electronics", "Clothing", "Other"])

data = pd.DataFrame([{{
    "Agent_Age": agent_age,
    "Agent_Rating": agent_rating,
    "Store_Latitude": store_lat,
    "Store_Longitude": store_lon,
    "Drop_Latitude": drop_lat,
    "Drop_Longitude": drop_lon,
    "Weather": weather,
    "Traffic": traffic,
    "Vehicle": vehicle,
    "Area": area,
    "Category": category
}}])

if st.sidebar.button("Predict Delivery Time"):
    pred = pipe.predict(data)[0]
    st.success(f"Estimated Delivery Time: {{pred:.2f}} (same units as dataset)")


