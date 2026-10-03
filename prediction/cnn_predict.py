import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "cnn",
    "deepmed_cnn_v2.keras"
)

IMG_SIZE = (224, 224)
THRESHOLD = 0.50

model = load_model(MODEL_PATH)


def predict_xray(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    raw_score = float(model.predict(image_array, verbose=0)[0][0])

    if raw_score >= THRESHOLD:
        prediction = "PNEUMONIA"
        confidence = raw_score
    else:
        prediction = "NORMAL"
        confidence = 1.0 - raw_score

    return {
        "prediction": prediction,
        "confidence": confidence * 100,
        "raw_score": raw_score
    }


if __name__ == "__main__":
    image_path = input("Enter X-ray image path: ").strip()

    if not os.path.exists(image_path):
        print("Image not found.")
    else:
        result = predict_xray(image_path)

        print("\n==============================")
        print("DeepMed AI - CNN Prediction")
        print("==============================")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.2f}%")
        print(f"Raw Score  : {result['raw_score']:.4f}")