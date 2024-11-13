import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import time
# Load the data
data = pd.read_csv('clean_data.csv')
test_data=pd.read_csv('MOCK_DATA.csv')
X = data.drop('diabetes', axis=1)
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train a new model on the resampled data
model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_resampled, y_resampled)

# Predictions and evaluation
y_pred_smote = model_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote))

# Save the model
with open('diabetes_model.pkl', 'wb') as file: 
      pickle.dump(model_smote, file)  

# Load the model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)
print(type(model))

# Load the trained model
start_time = time.time()
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)
print(f"Model loaded in {time.time() - start_time} seconds")
print(type(model))

# Collect user input for each feature
gender = input("Enter gender (Male/Female/Other): ")
age = int(input("Enter age: "))
hypertension = int(input("Hypertension (1 for Yes, 0 for No): "))
heart_disease = int(input("Heart disease (1 for Yes, 0 for No): "))
smoking_history = input("Smoking history (never/former/current/not current/unknown): ")
bmi = float(input("Enter BMI: "))
HbA1c_level = float(input("Enter HbA1c level: "))
blood_glucose_level = float(input("Enter blood glucose level: "))

# Encode 'gender' and 'smoking_history' inputs
gender_encoded = {'Male': 1, 'Female': 0, 'Other': 2}.get(gender)
smoking_encoded = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'unknown': 4}.get(smoking_history)

# Check if encoding was successful
if gender_encoded is None or smoking_encoded is None:
    print("Invalid input for gender or smoking history.")
else:
    # Time prediction step
    prediction_start_time = time.time()

    # Make prediction
    prediction = model.predict([[gender_encoded, age, hypertension, heart_disease, smoking_encoded, bmi, HbA1c_level, blood_glucose_level]])
    
    prediction_time = time.time() - prediction_start_time
    print(f"Prediction completed in {prediction_time} seconds")
    
    # Output result
    result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
    print(f"Prediction: {result}")
