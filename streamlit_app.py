import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

@st.cache_data
def load_csv_file():
    return pd.read_csv('https://raw.githubusercontent.com/ogrk/data/refs/heads/main/clean_data.csv')
data = load_csv_file()
def load_csv_file():
    return pd.read_csv('https://raw.githubusercontent.com/ogrk/data/refs/heads/main/MOCK_DATA.csv')    
test_data = load_csv_file()

X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_resampled, y_resampled)

with open('diabetes_model.pkl', 'wb') as file: 
    pickle.dump(model_smote, file)

@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
model = load_model()
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes status:")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension (1 for Yes, 0 for No)", [0, 1])
heart_disease = st.selectbox("Heart Disease (1 for Yes, 0 for No)", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "not current", "unknown"])
bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, value=5.5, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, value=100.0, step=1.0)

gender_encoded = {'Male': 1, 'Female': 0, 'Other': 2}.get(gender)
smoking_encoded = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'unknown': 4}.get(smoking_history)

if st.button("Predict"):
    if gender_encoded is None or smoking_encoded is None:
        st.error("Invalid input for gender or smoking history.")
    else:
        start_time = time.time()
        prediction = model.predict([[gender_encoded, age, hypertension, heart_disease, smoking_encoded, bmi, HbA1c_level, blood_glucose_level]])
        prediction_time = time.time() - start_time
        
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        st.write(f"Prediction: **{result}**")
        st.write(f"Prediction completed in {prediction_time:.4f} seconds")
