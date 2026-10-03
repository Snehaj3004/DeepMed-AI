import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "xray",
    "chest_xray"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn",
    "deepmed_cnn_final.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")
print()

print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42
)

class_names = test_ds.class_names

print("Classes:", class_names)
print()

# Check whether the model already contains normalization
print("First model layers:")

for layer in model.layers[:5]:
    print(
        layer.name,
        type(layer).__name__
    )

print()

y_true = []
y_scores = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    ).reshape(-1)

    y_true.extend(
        labels.numpy()
    )

    y_scores.extend(
        predictions
    )

y_true = np.array(y_true)
y_scores = np.array(y_scores)


print("========================================")
print("CNN TEST SET DIAGNOSTIC")
print("========================================")

print()
print("Total images:", len(y_true))

print()
print("NORMAL images:",
      np.sum(y_true == 0))

print("PNEUMONIA images:",
      np.sum(y_true == 1))

print()

print("Prediction score statistics")
print("----------------------------------------")

normal_scores = y_scores[y_true == 0]
pneumonia_scores = y_scores[y_true == 1]

print()
print("NORMAL images:")
print("Minimum :", normal_scores.min())
print("Maximum :", normal_scores.max())
print("Mean    :", normal_scores.mean())
print("Median  :", np.median(normal_scores))

print()
print("PNEUMONIA images:")
print("Minimum :", pneumonia_scores.min())
print("Maximum :", pneumonia_scores.max())
print("Mean    :", pneumonia_scores.mean())
print("Median  :", np.median(pneumonia_scores))

print()

print("========================================")
print("THRESHOLD ANALYSIS")
print("========================================")

thresholds = [
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
]

for threshold in thresholds:

    y_pred = (
        y_scores >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = (
        tn + tp
    ) / (
        tn + fp + fn + tp
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy={accuracy:.4f} | "
        f"Sensitivity={sensitivity:.4f} | "
        f"Specificity={specificity:.4f}"
    )

print()

print("========================================")
print("DEFAULT THRESHOLD = 0.50")
print("========================================")

y_pred = (
    y_scores >= 0.45
).astype(int)

print()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

print()

print(
    "ROC-AUC:",
    roc_auc_score(
        y_true,
        y_scores
    )
)

print()
print("========================================")
print("END OF DIAGNOSTIC")
print("========================================")