import os
import pickle
import numpy as np
import pandas as pd

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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = "datasets/patient_vitals/patient_vitals_v2.csv"

MODEL_DIR = "models/lstm"
RESULT_DIR = "results/lstm_v2"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepmed_lstm_v2.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepmed_lstm_v2_final.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v2_scaler.pkl"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# Prepare Patient Sequences
# ============================================================

features = [
    "heart_rate",
    "spo2",
    "temperature",
    "respiratory_rate"
]

TIME_STEPS = 24

X = []
y = []

patient_ids = df["patient_id"].unique()

for patient_id in patient_ids:

    patient_data = df[
        df["patient_id"] == patient_id
    ].sort_values("time_step")

    sequence = patient_data[features].values

    label = patient_data["status"].iloc[0]

    if len(sequence) == TIME_STEPS:

        X.append(sequence)
        y.append(label)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"\nSequence shape: {X.shape}")
print(f"Labels shape: {y.shape}")


# ============================================================
# Patient-Level Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print(f"Training patients: {len(X_train)}")
print(f"Testing patients: {len(X_test)}")


# ============================================================
# Scaling
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

X_test_2d = X_test.reshape(
    -1,
    X_test.shape[-1]
)

X_train_scaled = scaler.fit_transform(
    X_train_2d
)

X_test_scaled = scaler.transform(
    X_test_2d
)

X_train = X_train_scaled.reshape(
    X_train.shape
)

X_test = X_test_scaled.reshape(
    X_test.shape
)


with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"Scaler saved: {SCALER_PATH}")


# ============================================================
# Build LSTM Model
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(TIME_STEPS, len(features))
    ),

    Dropout(0.35),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.35),

    Dense(
        16,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


model.summary()


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
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# Train
# ============================================================

print("\nStarting LSTM v2 training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# Save Final Model
# ============================================================

model.save(FINAL_MODEL_PATH)

print(
    f"\nFinal model saved: {FINAL_MODEL_PATH}"
)


# ============================================================
# Save Training History
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

print(
    f"Training history saved: {HISTORY_PATH}"
)


# ============================================================
# Test Evaluation
# ============================================================

print("\nEvaluating on untouched test patients...")

probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

predictions = (
    probabilities >= 0.50
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# Specificity
# ============================================================

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("LSTM v2 TEST RESULTS")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "NORMAL",
            "ABNORMAL"
        ],
        zero_division=0
    )
)

print("=" * 60)