import os
import numpy as np
import pandas as pd

OUTPUT_PATH = "datasets/patient_vitals/patient_vitals_v2.csv"

np.random.seed(42)

NUM_PATIENTS = 1000
TIME_STEPS = 24

rows = []

for patient_id in range(1, NUM_PATIENTS + 1):

    status = np.random.choice([0, 1], p=[0.70, 0.30])

    # Patient-specific baseline
    hr_base = np.random.normal(78, 8)
    spo2_base = np.random.normal(97.2, 1.0)
    temp_base = np.random.normal(36.8, 0.25)
    rr_base = np.random.normal(16, 2)

    # Abnormal patients have only partially shifted values
    # so there is significant overlap with normal patients.
    if status == 1:
        hr_shift = np.random.normal(12, 7)
        spo2_shift = np.random.normal(-2.2, 1.2)
        temp_shift = np.random.normal(0.8, 0.35)
        rr_shift = np.random.normal(5, 2.5)
    else:
        hr_shift = 0
        spo2_shift = 0
        temp_shift = 0
        rr_shift = 0

    for time_step in range(TIME_STEPS):

        # Small gradual trend through the sequence
        trend = time_step / (TIME_STEPS - 1)

        heart_rate = (
            hr_base
            + hr_shift * trend
            + np.random.normal(0, 6)
        )

        spo2 = (
            spo2_base
            + spo2_shift * trend
            + np.random.normal(0, 0.8)
        )

        temperature = (
            temp_base
            + temp_shift * trend
            + np.random.normal(0, 0.18)
        )

        respiratory_rate = (
            rr_base
            + rr_shift * trend
            + np.random.normal(0, 2.0)
        )

        # Occasional natural fluctuations
        if np.random.random() < 0.05:
            heart_rate += np.random.normal(0, 12)

        if np.random.random() < 0.05:
            spo2 += np.random.normal(0, 1.5)

        if np.random.random() < 0.05:
            temperature += np.random.normal(0, 0.4)

        if np.random.random() < 0.05:
            respiratory_rate += np.random.normal(0, 4)

        # Keep values within realistic ranges
        heart_rate = np.clip(heart_rate, 45, 150)
        spo2 = np.clip(spo2, 88, 100)
        temperature = np.clip(temperature, 35.5, 40.0)
        respiratory_rate = np.clip(respiratory_rate, 8, 40)

        rows.append([
            patient_id,
            time_step,
            round(heart_rate, 2),
            round(spo2, 2),
            round(temperature, 2),
            round(respiratory_rate, 2),
            status
        ])


columns = [
    "patient_id",
    "time_step",
    "heart_rate",
    "spo2",
    "temperature",
    "respiratory_rate",
    "status"
]

df = pd.DataFrame(rows, columns=columns)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nDataset created successfully.")
print(f"File: {OUTPUT_PATH}")
print(f"Patients: {NUM_PATIENTS}")
print(f"Time steps per patient: {TIME_STEPS}")
print(f"Total rows: {len(df)}")

print("\nClass distribution:")
print(df.groupby("status")["patient_id"].nunique())

print("\nVital statistics:")
print(
    df[
        [
            "heart_rate",
            "spo2",
            "temperature",
            "respiratory_rate"
        ]
    ].describe()
)