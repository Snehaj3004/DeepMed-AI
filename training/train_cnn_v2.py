import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "cnn"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "cnn_v2"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42


print()
print("========================================")
print("DeepMed AI - CNN v2")
print("EfficientNetB0 Transfer Learning")
print("========================================")
print()

print(
    "Available GPUs:",
    tf.config.list_physical_devices("GPU")
)

print()


train_dir = os.path.join(
    DATASET_DIR,
    "train"
)

test_dir = os.path.join(
    DATASET_DIR,
    "test"
)


print("Loading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print()
print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print()
print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_ds.class_names

print()
print("Classes:", class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# CLASS WEIGHTS
# ------------------------------------------------------------

print()
print("Calculating class weights...")

train_labels = []

for _, labels in train_ds.unbatch().batch(1024):

    train_labels.extend(
        labels.numpy()
    )

train_labels = np.array(train_labels)

classes = np.unique(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    int(c): float(w)
    for c, w in zip(classes, weights)
}

print(
    "Class weights:",
    class_weights
)


# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.03
        ),

        tf.keras.layers.RandomZoom(
            0.08
        )
    ],
    name="data_augmentation"
)


# ------------------------------------------------------------
# EFFICIENTNETB0
# ------------------------------------------------------------

print()
print("Loading EfficientNetB0...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

base_model.trainable = False


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="xray_input"
)


x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(
    0.45
)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="prediction"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="DeepMed_CNN_v2"
)


# ------------------------------------------------------------
# COMPILE
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
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


print()
print("Model created.")
print()


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "deepmed_cnn_v2.keras"
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,

    monitor="val_loss",

    mode="min",

    save_best_only=True,

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",

    mode="min",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",

    mode="min",

    factor=0.5,

    patience=1,

    min_lr=1e-7,

    verbose=1
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print()
print("========================================")
print("STARTING CNN v2 TRAINING")
print("========================================")
print()


history = model.fit(
    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ------------------------------------------------------------
# LOAD BEST MODEL
# ------------------------------------------------------------

print()
print("Loading best CNN v2 model...")

model = tf.keras.models.load_model(
    model_path
)


# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

print()
print("========================================")
print("CNN v2 TEST EVALUATION")
print("========================================")
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


# ------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------

roc_auc = roc_auc_score(
    y_true,
    y_scores
)

print(
    f"Test ROC-AUC: {roc_auc:.4f}"
)


# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------

threshold = 0.50

y_pred = (
    y_scores >= threshold
).astype(int)


print()
print("Classification Report")
print("----------------------------------------")


report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("Confusion Matrix")
print("----------------------------------------")

print(cm)


plt.figure(
    figsize=(6, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    "DeepMed AI - CNN v2 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_true,
    y_scores
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "DeepMed AI - CNN v2 ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# ACCURACY GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN v2 Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# LOSS GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN v2 Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# SAVE FINAL MODEL
# ------------------------------------------------------------

final_model_path = os.path.join(
    MODEL_DIR,
    "deepmed_cnn_v2_final.keras"
)

model.save(
    final_model_path
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("========================================")
print("CNN v2 TRAINING COMPLETE")
print("========================================")

print()
print("Final model:")
print(final_model_path)

print()
print(
    f"Test ROC-AUC: {roc_auc:.4f}"
)

print()
print("Results:")
print(RESULTS_DIR)

print()
print("========================================")