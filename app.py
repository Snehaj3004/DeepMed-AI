import os
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from prediction.cnn_predict import predict_xray
from prediction.lstm_predict import predict_vitals
from prediction.anomaly_predict import predict_anomaly


app = Flask(__name__)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

LSTM_FEATURES = [
    "heart_rate",
    "spo2",
    "temperature",
    "respiratory_rate"
]

LSTM_TIME_STEPS = 24

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# Helper Functions
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def read_lstm_sequence(uploaded_file):

    try:
        frame = pd.read_csv(uploaded_file.stream)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError
    ) as error:
        raise ValueError(
            "Could not read the CSV file. Please upload a valid UTF-8 CSV."
        ) from error

    missing_features = [
        feature
        for feature in LSTM_FEATURES
        if feature not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing_features)
        )

    if len(frame) != LSTM_TIME_STEPS:
        raise ValueError(
            "The CSV must contain exactly 24 readings for one patient."
        )

    if "patient_id" in frame.columns:
        if (
            frame["patient_id"].isna().any()
            or frame["patient_id"].nunique() != 1
        ):
            raise ValueError(
                "The CSV must contain readings for exactly one patient."
            )

    if "time_step" in frame.columns:
        time_steps = pd.to_numeric(
            frame["time_step"],
            errors="coerce"
        )

        if time_steps.isna().any() or time_steps.duplicated().any():
            raise ValueError(
                "The time_step column must contain 24 unique numeric values."
            )

        ordered_steps = np.sort(
            time_steps.to_numpy()
        )

        if not np.allclose(
            np.diff(ordered_steps),
            np.diff(ordered_steps)[0]
        ):
            raise ValueError(
                "Readings must have consistent intervals between time steps."
            )

        frame = frame.assign(
            time_step=time_steps
        ).sort_values(
            "time_step",
        )

    vitals = frame[LSTM_FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(vitals).all():
        raise ValueError(
            "All vital measurements must be finite numeric values."
        )

    return vitals


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# CNN MODULE
# ============================================================

@app.route("/cnn")
def cnn():

    return render_template(
        "cnn.html"
    )


@app.route(
    "/cnn/predict",
    methods=["POST"]
)
def cnn_predict():

    if "xray" not in request.files:

        return render_template(
            "cnn.html",
            error="Please select an X-ray image."
        )

    file = request.files["xray"]

    if file.filename == "":

        return render_template(
            "cnn.html",
            error="Please select an X-ray image."
        )

    if not allowed_file(file.filename):

        return render_template(
            "cnn.html",
            error="Only PNG, JPG and JPEG images are allowed."
        )

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        result = predict_xray(
            filepath
        )

        return render_template(
            "result.html",
            result=result,
            image_filename=filename
        )

    except Exception as e:

        return render_template(
            "cnn.html",
            error=f"Prediction failed: {str(e)}"
        )


# ============================================================
# LSTM MODULE
# ============================================================

@app.route("/lstm")
def lstm():

    return render_template(
        "lstm.html"
    )


@app.route(
    "/lstm/predict",
    methods=["POST"]
)
def lstm_predict():

    uploaded_file = request.files.get("vitals_csv")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template(
            "lstm.html",
            error="Please upload a CSV containing 24 patient readings."
        )

    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template(
            "lstm.html",
            error="Please upload a CSV file."
        )

    try:
        vitals = read_lstm_sequence(uploaded_file)
        result = predict_vitals(
            vitals
        )

        return render_template(
            "lstm_result.html",
            result=result
        )

    except ValueError as error:

        return render_template(
            "lstm.html",
            error=str(error)
        )


    except Exception as e:

        return render_template(
            "lstm.html",
            error=f"Prediction failed: {str(e)}"
        )


# ============================================================
# AUTOENCODER ANOMALY DETECTION MODULE
# ============================================================

@app.route("/anomaly")
def anomaly():

    return render_template(
        "anomaly.html"
    )


@app.route(
    "/anomaly/predict",
    methods=["POST"]
)
def anomaly_predict():

    try:
        measurements = [
            float(request.form[field])
            for field in (
                "heart_rate",
                "bp_systolic",
                "bp_diastolic",
                "spo2",
                "respiration_rate",
                "body_temperature",
                "blood_glucose"
            )
        ]
    except KeyError:
        return render_template(
            "anomaly.html",
            error="Please enter all required measurements."
        )
    except ValueError:
        return render_template(
            "anomaly.html",
            error="Please enter valid numeric values."
        )

    try:
        result = predict_anomaly(*measurements)

        return render_template(
            "anomaly.html",
            result=result
        )
    except ValueError as error:
        return render_template(
            "anomaly.html",
            error=str(error)
        )

    except Exception as e:

        return render_template(
            "anomaly.html",
            error=f"Prediction failed: {str(e)}"
        )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )