import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight


# ==============================
# Configuration
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "xray",
    "chest_xray"
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models", "cnn")
RESULT_DIR = os.path.join(BASE_DIR, "results", "cnn")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42


print("=" * 60)
print("DeepMed AI - CNN Training")
print("=" * 60)

print(f"\nDataset path:")
print(DATASET_DIR)

print(f"\nTraining directory:")
print(TRAIN_DIR)

print(f"\nTest directory:")
print(TEST_DIR)


# ==============================
# Check dataset
# ==============================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ==============================
# Load training dataset
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.20,
    subset="training"
)


# ==============================
# Load validation dataset
# ==============================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.20,
    subset="validation"
)


# ==============================
# Load test dataset
# ==============================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)


# ==============================
# Dataset optimization
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# ==============================
# Calculate class weights
# ==============================

train_labels = []

for _, labels in train_dataset.unbatch():
    train_labels.append(int(labels.numpy()[0]))

train_labels = np.array(train_labels)

classes = np.unique(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    int(classes[i]): float(weights[i])
    for i in range(len(classes))
}

print("\nClass weights:")
print(class_weights)


# ==============================
# Data augmentation
# ==============================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(
            0.05
        ),
        tf.keras.layers.RandomZoom(
            0.10
        )
    ],
    name="data_augmentation"
)


# ==============================
# CNN Model
# ==============================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        data_augmentation,

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ]
)


# ==============================
# Compile model
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ==============================
# Display model
# ==============================

print("\n")
model.summary()


# ==============================
# Callbacks
# ==============================

model_path = os.path.join(
    MODEL_DIR,
    "deepmed_cnn.keras"
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ==============================
# Train
# ==============================

print("\n")
print("=" * 60)
print("Starting CNN Training")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ==============================
# Save final model
# ==============================

final_model_path = os.path.join(
    MODEL_DIR,
    "deepmed_cnn_final.keras"
)

model.save(final_model_path)

print("\nFinal model saved:")
print(final_model_path)


# ==============================
# Training graphs
# ==============================

history_dict = history.history


# Accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    history_dict["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_dict["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    RESULT_DIR,
    "accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Loss
plt.figure(figsize=(10, 6))

plt.plot(
    history_dict["loss"],
    label="Training Loss"
)

plt.plot(
    history_dict["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    RESULT_DIR,
    "loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# Test evaluation
# ==============================

print("\n")
print("=" * 60)
print("Evaluating CNN on Test Dataset")
print("=" * 60)

test_results = model.evaluate(
    test_dataset,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    test_results
):
    print(f"{name}: {value:.4f}")


# ==============================
# Predictions
# ==============================

y_true = []
y_prob = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten()
    )

    y_prob.extend(
        predictions.flatten()
    )


y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (
    y_prob >= 0.5
).astype(int)


# ==============================
# Classification report
# ==============================

print("\n")
print("=" * 60)
print("Classification Report")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

report_path = os.path.join(
    RESULT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(report)


# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    values_format="d"
)

plt.title("CNN Confusion Matrix")

cm_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# ROC-AUC
# ==============================

auc_score = roc_auc_score(
    y_true,
    y_prob
)

print(f"\nROC-AUC: {auc_score:.4f}")


fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("CNN ROC Curve")

plt.legend()
plt.grid(True)

roc_path = os.path.join(
    RESULT_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# Final information
# ==============================

print("\n")
print("=" * 60)
print("CNN TRAINING COMPLETED")
print("=" * 60)

print("\nModel:")
print(final_model_path)

print("\nBest checkpoint:")
print(model_path)

print("\nResults:")
print(RESULT_DIR)

print("\nROC-AUC:")
print(f"{auc_score:.4f}")