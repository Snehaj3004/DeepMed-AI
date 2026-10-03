import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = (
    "datasets/anomaly/"
    "synthetic_health_dataset_stratified_20_patients.csv"
)

MODEL_DIR = "models/autoencoder"
RESULT_DIR = "results/autoencoder"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepmed_autoencoder.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepmed_autoencoder_final.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "autoencoder_scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "anomaly_threshold.pkl"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# Features
# ============================================================

FEATURES = [
    "Heart_Rate",
    "BP_Systolic",
    "BP_Diastolic",
    "SpO2",
    "Respiration_Rate",
    "Body_Temperature",
    "Blood_Glucose"
]


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading health dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# Check Required Columns
# ============================================================

missing_columns = [
    column
    for column in FEATURES + ["Is_Anomaly", "Patient_ID"]
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# Convert Features to Numeric
# ============================================================

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df["Is_Anomaly"] = pd.to_numeric(
    df["Is_Anomaly"],
    errors="coerce"
)


# ============================================================
# Display Dataset Information
# ============================================================

print("\nAnomaly distribution:")

print(
    df["Is_Anomaly"].value_counts(
        dropna=False
    )
)


print("\nMissing values:")

print(
    df[FEATURES].isna().sum()
)


# ============================================================
# Remove rows with missing required measurements
# ============================================================

df_clean = df.dropna(
    subset=FEATURES + ["Is_Anomaly"]
).copy()


print(
    "\nRows after removing incomplete "
    f"measurements: {len(df_clean)}"
)


# ============================================================
# Separate NORMAL and ANOMALY records
# ============================================================

normal_df = df_clean[
    df_clean["Is_Anomaly"] == 0
].copy()

anomaly_df = df_clean[
    df_clean["Is_Anomaly"] == 1
].copy()


print(
    f"\nNormal records : {len(normal_df)}"
)

print(
    f"Anomaly records: {len(anomaly_df)}"
)


# ============================================================
# Patient-Level Split
# ============================================================

normal_patients = normal_df[
    "Patient_ID"
].unique()


train_patients, validation_patients = train_test_split(
    normal_patients,
    test_size=0.20,
    random_state=42
)


train_normal = normal_df[
    normal_df["Patient_ID"].isin(
        train_patients
    )
].copy()


validation_normal = normal_df[
    normal_df["Patient_ID"].isin(
        validation_patients
    )
].copy()


print(
    "\nPatient-level split:"
)

print(
    f"Training normal patients   : "
    f"{len(train_patients)}"
)

print(
    f"Validation normal patients : "
    f"{len(validation_patients)}"
)

print(
    f"Training normal records    : "
    f"{len(train_normal)}"
)

print(
    f"Validation normal records  : "
    f"{len(validation_normal)}"
)


# ============================================================
# Scaling
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    train_normal[FEATURES]
)

X_validation = scaler.transform(
    validation_normal[FEATURES]
)

X_anomaly = scaler.transform(
    anomaly_df[FEATURES]
)


with open(
    SCALER_PATH,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    f"Scaler saved: {SCALER_PATH}"
)


# ============================================================
# Autoencoder Architecture
# ============================================================

print("\nBuilding Autoencoder...")

input_dim = len(FEATURES)

input_layer = Input(
    shape=(input_dim,)
)


# Encoder

encoded = Dense(
    16,
    activation="relu"
)(input_layer)

encoded = Dropout(
    0.10
)(encoded)

encoded = Dense(
    8,
    activation="relu"
)(encoded)


# Decoder

decoded = Dense(
    16,
    activation="relu"
)(encoded)

decoded = Dense(
    input_dim,
    activation="linear"
)(decoded)


autoencoder = Model(
    input_layer,
    decoded
)


autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


autoencoder.summary()


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# Train Autoencoder
# ============================================================

print(
    "\nStarting Autoencoder training...\n"
)


history = autoencoder.fit(

    X_train,

    X_train,

    validation_data=(
        X_validation,
        X_validation
    ),

    epochs=50,

    batch_size=32,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# Save Final Model
# ============================================================

autoencoder.save(
    FINAL_MODEL_PATH
)


print(
    f"\nFinal model saved: "
    f"{FINAL_MODEL_PATH}"
)


# ============================================================
# Reconstruction Errors
# ============================================================

print(
    "\nCalculating reconstruction errors..."
)


normal_reconstructed = autoencoder.predict(
    X_validation,
    verbose=0
)


anomaly_reconstructed = autoencoder.predict(
    X_anomaly,
    verbose=0
)


normal_errors = np.mean(
    np.square(
        X_validation -
        normal_reconstructed
    ),
    axis=1
)


anomaly_errors = np.mean(
    np.square(
        X_anomaly -
        anomaly_reconstructed
    ),
    axis=1
)


# ============================================================
# Determine Threshold
# ============================================================

threshold = np.percentile(
    normal_errors,
    95
)


with open(
    THRESHOLD_PATH,
    "wb"
) as f:

    pickle.dump(
        threshold,
        f
    )


print(
    f"\nAnomaly threshold: "
    f"{threshold:.6f}"
)


# ============================================================
# Create Evaluation Dataset
# ============================================================

evaluation_errors = np.concatenate(
    [
        normal_errors,
        anomaly_errors
    ]
)


y_true = np.concatenate(
    [
        np.zeros(
            len(normal_errors)
        ),
        np.ones(
            len(anomaly_errors)
        )
    ]
)


y_pred = (
    evaluation_errors >= threshold
).astype(int)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    evaluation_errors
)


cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# Specificity
# ============================================================

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)

print(
    "DEEPmed AI AUTOENCODER RESULTS"
)

print("=" * 60)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"F1 Score    : {f1:.4f}"
)

print(
    f"ROC-AUC     : {auc:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(cm)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "ANOMALY"
        ],
        zero_division=0
    )
)


# ============================================================
# Reconstruction Error Statistics
# ============================================================

print(
    "\nReconstruction Error Statistics:"
)

print(
    f"Normal mean   : "
    f"{normal_errors.mean():.6f}"
)

print(
    f"Normal median : "
    f"{np.median(normal_errors):.6f}"
)

print(
    f"Normal max    : "
    f"{normal_errors.max():.6f}"
)

print(
    f"Anomaly mean  : "
    f"{anomaly_errors.mean():.6f}"
)

print(
    f"Anomaly median: "
    f"{np.median(anomaly_errors):.6f}"
)

print(
    f"Anomaly max   : "
    f"{anomaly_errors.max():.6f}"
)

print("=" * 60)