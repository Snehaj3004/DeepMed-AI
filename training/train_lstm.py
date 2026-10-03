import os
import joblib
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "patient_vitals",
    "patient_vitals.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "lstm"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "lstm"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TIME_STEPS = 24

FEATURES = [
    "heart_rate",
    "spo2",
    "temperature",
    "respiratory_rate"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


print("=" * 60)
print("DEEPMED AI - LSTM TRAINING")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading patient vitals dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df)}")
print(f"Total patients: {df['patient_id'].nunique()}")

print("\nDataset columns:")
print(list(df.columns))


# ============================================================
# CREATE PATIENT-LEVEL SEQUENCES
# ============================================================

print("\nCreating patient time-series sequences...")

X = []
y = []
patient_ids = []

for patient_id, patient_data in df.groupby("patient_id"):

    patient_data = patient_data.sort_values("time_step")

    values = patient_data[FEATURES].values

    label = patient_data["status"].iloc[-1]

    if len(values) == TIME_STEPS:

        X.append(values)
        y.append(label)
        patient_ids.append(patient_id)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
patient_ids = np.array(patient_ids)


print(f"\nSequences created: {len(X)}")
print(f"Sequence shape: {X.shape}")

print("\nClass distribution:")
print(
    pd.Series(y).value_counts().sort_index()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting patients into training and testing sets...")

train_ids, test_ids = train_test_split(
    np.arange(len(X)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train = X[train_ids]
X_test = X[test_ids]

y_train = y[train_ids]
y_test = y[test_ids]


print(f"\nTraining sequences: {len(X_train)}")
print(f"Testing sequences : {len(X_test)}")


# ============================================================
# FEATURE SCALING
# ============================================================

print("\nScaling vital-sign features...")

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    len(FEATURES)
)

X_test_2d = X_test.reshape(
    -1,
    len(FEATURES)
)

scaler.fit(X_train_2d)

X_train_scaled = scaler.transform(
    X_train_2d
).reshape(
    X_train.shape
)

X_test_scaled = scaler.transform(
    X_test_2d
).reshape(
    X_test.shape
)


SCALER_PATH = os.path.join(
    MODEL_DIR,
    "lstm_scaler.pkl"
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print(f"Scaler saved to:")
print(SCALER_PATH)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential([
    LSTM(
        64,
        input_shape=(
            TIME_STEPS,
            len(FEATURES)
        ),
        return_sequences=True
    ),

    Dropout(0.30),

    LSTM(
        32
    ),

    Dropout(0.30),

    Dense(
        16,
        activation="relu"
    ),

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
        "accuracy",
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
# CALLBACKS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepmed_lstm.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepmed_lstm_final.keras"
)


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting LSTM training...")

history = model.fit(
    X_train_scaled,
    y_train,

    validation_split=0.20,

    epochs=30,
    batch_size=32,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best LSTM model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_scores = best_model.predict(
    X_test_scaled,
    verbose=0
).flatten()

y_pred = (
    y_scores >= 0.50
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_scores
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

specificity = tn / (tn + fp)


print("\n")
print("=" * 60)
print("LSTM TEST EVALUATION")
print("=" * 60)

print(f"\nAccuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")


print("\nClassification Report")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NORMAL",
            "ABNORMAL"
        ],
        digits=4
    )
)


print("\nConfusion Matrix")
print("-" * 60)

print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    FINAL_MODEL_PATH
)

print("\nFinal model saved to:")
print(FINAL_MODEL_PATH)

print("\nBest model saved to:")
print(MODEL_PATH)

print("\nScaler saved to:")
print(SCALER_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_path = os.path.join(
    RESULTS_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print("\nTraining history saved to:")
print(history_path)


print("\n" + "=" * 60)
print("LSTM TRAINING COMPLETE")
print("=" * 60)