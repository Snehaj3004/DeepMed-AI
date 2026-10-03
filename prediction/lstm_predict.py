import os
import pickle
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/lstm/deepmed_lstm_v2.keras"
SCALER_PATH = "models/lstm/lstm_v2_scaler.pkl"

TIME_STEPS = 24

FEATURES = [
    "heart_rate",
    "spo2",
    "temperature",
    "respiratory_rate"
]


print("Loading LSTM v2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("LSTM v2 model loaded successfully.")


def predict_vitals(vitals):

    vitals = np.array(
        vitals,
        dtype=np.float32
    )

    if vitals.shape != (TIME_STEPS, 4):
        raise ValueError(
            "Input must contain exactly 24 rows "
            "with 4 values per row: "
            "heart_rate, spo2, temperature, respiratory_rate"
        )

    # Scale using the same scaler used during training
    vitals_scaled = scaler.transform(vitals)

    # Add batch dimension
    vitals_scaled = np.expand_dims(
        vitals_scaled,
        axis=0
    )

    raw_score = float(
        model.predict(
            vitals_scaled,
            verbose=0
        )[0][0]
    )

    if raw_score >= 0.50:
        prediction = "ABNORMAL"
        confidence = raw_score
    else:
        prediction = "NORMAL"
        confidence = 1 - raw_score

    return {
        "prediction": prediction,
        "confidence": confidence,
        "raw_score": raw_score
    }


# ============================================================
# Manual Testing
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DeepMed AI - LSTM v2 Patient Vital Prediction")
    print("=" * 60)

    print("\nEnter 24 rows of patient vitals.")

    print(
        "\nFormat:"
        "\nHeart Rate, SpO2, Temperature, Respiratory Rate"
    )

    print(
        "\nExample:"
        "\n78, 98, 36.8, 16"
    )

    print("\n" + "-" * 60)

    vitals = []

    for i in range(TIME_STEPS):

        while True:

            try:

                values = input(
                    f"Step {i + 1:02d}: "
                ).strip()

                values = [
                    float(x.strip())
                    for x in values.split(",")
                ]

                if len(values) != 4:
                    print(
                        "Please enter exactly 4 values."
                    )
                    continue

                vitals.append(values)

                break

            except ValueError:

                print(
                    "Invalid input. "
                    "Use numbers separated by commas."
                )

    result = predict_vitals(vitals)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Raw Score  : "
        f"{result['raw_score']:.4f}"
    )

    print("=" * 60)