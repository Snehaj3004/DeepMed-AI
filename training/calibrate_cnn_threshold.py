import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
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


print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = val_ds.class_names

print("Classes:", class_names)
print()


y_true = []
y_scores = []


print("Generating validation predictions...")

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    ).reshape(-1)

    y_true.extend(labels.numpy())
    y_scores.extend(predictions)


y_true = np.array(y_true)
y_scores = np.array(y_scores)


print()
print("========================================")
print("VALIDATION SET")
print("========================================")

print("Total images:", len(y_true))
print("NORMAL:", np.sum(y_true == 0))
print("PNEUMONIA:", np.sum(y_true == 1))

print()

print(
    "Validation ROC-AUC:",
    roc_auc_score(y_true, y_scores)
)

print()
print("========================================")
print("THRESHOLD SEARCH")
print("========================================")


thresholds = np.arange(
    0.30,
    0.81,
    0.01
)

results = []


for threshold in thresholds:

    y_pred = (
        y_scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

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

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    })


best = max(
    results,
    key=lambda x: x["balanced_accuracy"]
)


print()
print("Best threshold based on validation")
print("----------------------------------------")

print(
    f"Threshold         : {best['threshold']:.2f}"
)

print(
    f"Accuracy           : {best['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy  : {best['balanced_accuracy']:.4f}"
)

print(
    f"Sensitivity        : {best['sensitivity']:.4f}"
)

print(
    f"Specificity        : {best['specificity']:.4f}"
)


print()
print("========================================")
print("ALL THRESHOLD RESULTS")
print("========================================")

for result in results:

    print(
        f"{result['threshold']:.2f} | "
        f"Accuracy={result['accuracy']:.4f} | "
        f"Balanced={result['balanced_accuracy']:.4f} | "
        f"Sensitivity={result['sensitivity']:.4f} | "
        f"Specificity={result['specificity']:.4f}"
    )


print()
print("========================================")
print("RECOMMENDED THRESHOLD")
print("========================================")

print(
    f"{best['threshold']:.2f}"
)