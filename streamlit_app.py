import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

# Title and information
st.title('Diabetes Prediction App')
st.info('This app uses a machine learning model to predict diabetes')

# Load the dataset
data_url = 'https://raw.githubusercontent.com/ogrk/data/refs/heads/main/clean_data.csv'
test_data_url = 'https://raw.githubusercontent.com/ogrk/data/refs/heads/main/MOCK_DATA.csv'
data = pd.read_csv(data_url)
test_data = pd.read_csv(test_data_url)

# Prepare the data
X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train a model on the resampled data
model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_resampled, y_resampled)

# Save and load the model
joblib.dump(model_smote, 'diabetes_model.joblib')
model = joblib.load('diabetes_model.joblib')

# User input fields
st.header("Input Patient Data")
gender = st.selectbox("Select gender", options=['Male', 'Female', 'Other'])
age = st.number_input("Enter age", min_value=0, max_value=120, step=1)
hypertension = st.selectbox("Hypertension (1 for Yes, 0 for No)", options=[0, 1])
heart_disease = st.selectbox("Heart disease (1 for Yes, 0 for No)", options=[0, 1])
smoking_history = st.selectbox("Smoking history", options=['never', 'former', 'current', 'not current', 'unknown'])
bmi = st.number_input("Enter BMI", min_value=0.0, max_value=100.0, step=0.1)
HbA1c_level = st.number_input("Enter HbA1c level", min_value=0.0, max_value=20.0, step=0.1)
blood_glucose_level = st.number_input("Enter blood glucose level", min_value=0.0, max_value=500.0, step=0.1)

# Encode 'gender' and 'smoking_history' inputs
gender_encoded = {'Male': 1, 'Female': 0, 'Other': 2}.get(gender)
smoking_encoded = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'unknown': 4}.get(smoking_history)

# Button to make prediction
if st.button("Predict"):
    if gender_encoded is None or smoking_encoded is None:
        st.error("Invalid input for gender or smoking history.")
    else:
        start_time = time.time()
        prediction = model.predict([[gender_encoded, age, hypertension, heart_disease, smoking_encoded, bmi, HbA1c_level, blood_glucose_level]])
        prediction_time = time.time() - start_time
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        
        # Display result
        st.write(f"Prediction: {result}")
        st.write(f"Prediction completed in {prediction_time:.2f} seconds")
