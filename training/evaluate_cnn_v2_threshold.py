import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "xray",
    "chest_xray",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn",
    "deepmed_cnn_v2.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

THRESHOLD = 0.21


print("=" * 50)
print("CNN v2 TEST EVALUATION - THRESHOLD 0.21")
print("=" * 50)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(test_ds.class_names)

print("\nLoading CNN v2 model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# Generate predictions
y_true = []
y_scores = []


print("\nGenerating predictions...")

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(labels.numpy())
    y_scores.extend(predictions.flatten())


y_true = np.array(y_true)
y_scores = np.array(y_scores)


# Apply calibrated threshold
y_pred = (
    y_scores >= THRESHOLD
).astype(int)


# Metrics
accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_scores
)


# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()


specificity = tn / (tn + fp)

sensitivity = tp / (tp + fn)


print("\n")
print("=" * 50)
print("CNN v2 FINAL TEST RESULTS")
print("=" * 50)

print(f"\nThreshold          : {THRESHOLD:.2f}")
print(f"Test Accuracy     : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"ROC-AUC           : {roc_auc:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Sensitivity       : {sensitivity:.4f}")
print(f"Specificity       : {specificity:.4f}")
print(f"F1 Score          : {f1:.4f}")


print("\n")
print("Classification Report")
print("-" * 50)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ],
        digits=4
    )
)


print("\n")
print("Confusion Matrix")
print("-" * 50)

print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)


print("\n")
print("=" * 50)
print("INTERPRETATION")
print("=" * 50)

print(f"\nCorrect NORMAL predictions    : {tn}")
print(f"Incorrect NORMAL predictions  : {fp}")

print(f"Correct PNEUMONIA predictions : {tp}")
print(f"Missed PNEUMONIA cases        : {fn}")

print("\nEvaluation complete.")