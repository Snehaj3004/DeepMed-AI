import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "xray",
    "chest_xray",
    "train"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn",
    "deepmed_cnn_v2.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


print("=" * 50)
print("CNN v2 THRESHOLD CALIBRATION")
print("=" * 50)

print("\nLoading validation dataset...")

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = validation_ds.class_names

print("\nClasses:")
print(class_names)

print("\nLoading CNN v2 model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# Get true labels and predictions
y_true = []
y_scores = []

for images, labels in validation_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_scores.extend(predictions.flatten())


y_true = np.array(y_true)
y_scores = np.array(y_scores)

print("\nValidation samples:", len(y_true))

print("NORMAL samples:", np.sum(y_true == 0))
print("PNEUMONIA samples:", np.sum(y_true == 1))


# ROC-AUC
auc = roc_auc_score(y_true, y_scores)

print("\nValidation ROC-AUC:")
print(f"{auc:.4f}")


# Find best threshold using balanced accuracy
thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = None
best_balanced_accuracy = -1

results = []

for threshold in thresholds:

    y_pred = (y_scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results.append(
        (
            threshold,
            accuracy,
            balanced_accuracy,
            sensitivity,
            specificity
        )
    )

    if balanced_accuracy > best_balanced_accuracy:

        best_balanced_accuracy = balanced_accuracy
        best_threshold = threshold


print("\n" + "=" * 50)
print("THRESHOLD ANALYSIS")
print("=" * 50)

print(
    "\nThreshold | Accuracy | Balanced Acc | Sensitivity | Specificity"
)

for threshold, accuracy, balanced_accuracy, sensitivity, specificity in results:

    if abs(threshold - best_threshold) < 0.001 or threshold in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:

        print(
            f"{threshold:8.2f} | "
            f"{accuracy:8.4f} | "
            f"{balanced_accuracy:12.4f} | "
            f"{sensitivity:11.4f} | "
            f"{specificity:10.4f}"
        )


# Final best threshold
y_pred_best = (y_scores >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred_best
).ravel()

accuracy = accuracy_score(
    y_true,
    y_pred_best
)

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)


print("\n" + "=" * 50)
print("BEST CNN v2 THRESHOLD")
print("=" * 50)

print(f"\nBest threshold      : {best_threshold:.2f}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Balanced accuracy   : {best_balanced_accuracy:.4f}")
print(f"Sensitivity/Recall  : {sensitivity:.4f}")
print(f"Specificity         : {specificity:.4f}")

print("\nConfusion Matrix:")
print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)

print("\nUse this threshold in:")
print("prediction/cnn_predict.py")

print("\nCalibration complete.")