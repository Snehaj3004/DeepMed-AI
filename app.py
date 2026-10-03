import os
import numpy as np

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

    try:

        # ----------------------------------------------------
        # Get current patient vitals
        # ----------------------------------------------------

        heart_rate = float(
            request.form["heart_rate"]
        )

        spo2 = float(
            request.form["spo2"]
        )

        temperature = float(
            request.form["temperature"]
        )

        respiratory_rate = float(
            request.form["respiratory_rate"]
        )


        # ----------------------------------------------------
        # Get selected pattern
        # ----------------------------------------------------

        pattern = request.form.get(
            "pattern",
            "normal"
        )


        # ----------------------------------------------------
        # Base vital-sign values
        # ----------------------------------------------------

        base = np.array(
            [
                heart_rate,
                spo2,
                temperature,
                respiratory_rate
            ],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Generate 24-step sequence
        # ----------------------------------------------------

        vitals = []


        for i in range(24):

            progress = i / 23


            # =================================================
            # Normal Pattern
            # =================================================

            if pattern == "normal":

                values = base.copy()

                values[0] += np.random.normal(
                    0,
                    2.5
                )

                values[1] += np.random.normal(
                    0,
                    0.3
                )

                values[2] += np.random.normal(
                    0,
                    0.08
                )

                values[3] += np.random.normal(
                    0,
                    0.8
                )


            # =================================================
            # Abnormal Pattern
            # =================================================

            elif pattern == "abnormal":

                values = base.copy()

                values[0] += (
                    12 * progress
                )

                values[1] -= (
                    2 * progress
                )

                values[2] += (
                    0.8 * progress
                )

                values[3] += (
                    5 * progress
                )


                values[0] += np.random.normal(
                    0,
                    3
                )

                values[1] += np.random.normal(
                    0,
                    0.4
                )

                values[2] += np.random.normal(
                    0,
                    0.10
                )

                values[3] += np.random.normal(
                    0,
                    1
                )


            # =================================================
            # Natural Variation
            # =================================================

            else:

                values = base.copy()

                values[0] += np.random.normal(
                    0,
                    4
                )

                values[1] += np.random.normal(
                    0,
                    0.5
                )

                values[2] += np.random.normal(
                    0,
                    0.12
                )

                values[3] += np.random.normal(
                    0,
                    1.2
                )


            # ------------------------------------------------
            # Keep values within reasonable ranges
            # ------------------------------------------------

            values[0] = np.clip(
                values[0],
                45,
                150
            )

            values[1] = np.clip(
                values[1],
                88,
                100
            )

            values[2] = np.clip(
                values[2],
                35.5,
                40
            )

            values[3] = np.clip(
                values[3],
                8,
                40
            )


            vitals.append(
                values.tolist()
            )


        # ----------------------------------------------------
        # LSTM Prediction
        # ----------------------------------------------------

        result = predict_vitals(
            vitals
        )


        # ----------------------------------------------------
        # Display Result
        # ----------------------------------------------------

        return render_template(
            "lstm_result.html",
            result=result
        )


    except KeyError:

        return render_template(
            "lstm.html",
            error="Please enter all required vital signs."
        )


    except ValueError:

        return render_template(
            "lstm.html",
            error="Please enter valid numeric values."
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

        # ----------------------------------------------------
        # Get patient measurements
        # ----------------------------------------------------

        heart_rate = float(
            request.form["heart_rate"]
        )

        bp_systolic = float(
            request.form["bp_systolic"]
        )

        bp_diastolic = float(
            request.form["bp_diastolic"]
        )

        spo2 = float(
            request.form["spo2"]
        )

        respiration_rate = float(
            request.form["respiration_rate"]
        )

        body_temperature = float(
            request.form["body_temperature"]
        )

        blood_glucose = float(
            request.form["blood_glucose"]
        )


        # ----------------------------------------------------
        # Autoencoder Prediction
        # ----------------------------------------------------

        result = predict_anomaly(
            heart_rate,
            bp_systolic,
            bp_diastolic,
            spo2,
            respiration_rate,
            body_temperature,
            blood_glucose
        )


        # ----------------------------------------------------
        # Display Result
        # ----------------------------------------------------

        return render_template(
            "anomaly.html",
            result=result
        )


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