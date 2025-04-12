import pickle
import numpy as np

# Load trained model
with open("app/artifacts/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("app/artifacts/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Optional: load scaler if used
try:
    with open("app/artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None

def preprocess_input(data: dict):
    # Encode categorical fields
    for col, encoder in label_encoders.items():
        if col in data:
            data[col] = encoder.transform([data[col]])[0]

    # Feature ordering must match model training
    ordered_keys = [
        'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure',
        'Heart Rate', 'Daily Steps'
    ]

    features = [data[key] for key in ordered_keys]

    if scaler:
        features = scaler.transform([features])[0]

    return np.array([features])

def predict(data: dict):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return prediction
