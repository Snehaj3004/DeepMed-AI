import os
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "autoencoder", "deepmed_autoencoder_final.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "autoencoder", "autoencoder_scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR, "models", "autoencoder", "anomaly_threshold.pkl"
)


FEATURES = [
    "Heart_Rate",
    "BP_Systolic",
    "BP_Diastolic",
    "SpO2",
    "Respiration_Rate",
    "Body_Temperature",
    "Blood_Glucose"
]


model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)


def predict_anomaly(
    heart_rate,
    bp_systolic,
    bp_diastolic,
    spo2,
    respiration_rate,
    body_temperature,
    blood_glucose
):

    values = pd.DataFrame([[
    heart_rate,
    bp_systolic,
    bp_diastolic,
    spo2,
    respiration_rate,
    body_temperature,
    blood_glucose
    ]], columns=FEATURES)

    scaled_values = scaler.transform(values)

    reconstructed = model.predict(
        scaled_values,
        verbose=0
    )

    reconstruction_error = np.mean(
        np.square(scaled_values - reconstructed),
        axis=1
    )[0]

    if reconstruction_error >= threshold:
        status = "ANOMALY"
    else:
        status = "NORMAL"

    return {
        "status": status,
        "reconstruction_error": float(reconstruction_error),
        "threshold": float(threshold)
    }


if __name__ == "__main__":

    print("=" * 60)
    print("DEEPmed AI - AUTOENCODER ANOMALY PREDICTION")
    print("=" * 60)

    print("\nEnter patient vital measurements:\n")

    heart_rate = float(input("Heart Rate (bpm): "))
    bp_systolic = float(input("BP Systolic (mmHg): "))
    bp_diastolic = float(input("BP Diastolic (mmHg): "))
    spo2 = float(input("SpO2 (%): "))
    respiration_rate = float(input("Respiration Rate (breaths/min): "))
    body_temperature = float(input("Body Temperature (°F): "))
    blood_glucose = float(input("Blood Glucose: "))

    result = predict_anomaly(
        heart_rate,
        bp_systolic,
        bp_diastolic,
        spo2,
        respiration_rate,
        body_temperature,
        blood_glucose
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Status               : {result['status']}")
    print(f"Reconstruction Error : {result['reconstruction_error']:.6f}")
    print(f"Anomaly Threshold    : {result['threshold']:.6f}")

    if result["status"] == "ANOMALY":
        print("\n⚠ Anomalous pattern detected.")
    else:
        print("\n✓ Patient measurements appear normal.")

    print("=" * 60)