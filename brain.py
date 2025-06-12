import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os


def save_model(data_path, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])
    y = data['Diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, model_path)
    
    return acc

def predict(new_data, model_path):
    model = joblib.load(model_path)
    
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    prediction = model.predict(new_data)
    return prediction[0]

def accuracy(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])
    y = data['Diagnosis']
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)
