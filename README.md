# DeepMed AI

DeepMed AI is a Flask-based medical AI web application for clinical prediction and screening. It combines multiple trained deep learning models to support:

- Chest X-ray classification for pneumonia vs. normal
- Patient vital-sign sequence analysis for abnormal health trends
- Autoencoder-based anomaly detection for patient measurements

This project is designed as a practical demo and research prototype for AI-assisted medical decision support.

## Features

### 1. Chest X-ray detection
- Upload a chest X-ray image
- CNN model predicts whether the image is `NORMAL` or `PNEUMONIA`
- Confidence score and raw prediction score are shown in the UI

### 2. LSTM vital-sign analysis
- Input a patient’s heart rate, SpO2, temperature, and respiratory rate
- The app generates a 24-step sequence pattern to simulate monitoring over time
- LSTM model classifies the trend as `NORMAL` or `ABNORMAL`

### 3. Autoencoder anomaly detection
- Input multiple patient vital parameters
- Autoencoder reconstructs the data and flags anomalies using a learned threshold
- Output includes reconstruction error and anomaly status

## Tech stack

- Python
- Flask
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn
- Pillow for image handling
- HTML / CSS / JavaScript for the frontend

## Project structure

```text
DeepMed AI/
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies
├── datasets/                  # Data used for training and testing
├── models/                    # Trained ML models and related artifacts
│   ├── autoencoder/
│   ├── cnn/
│   └── lstm/
├── prediction/                # Prediction scripts for each model
│   ├── anomaly_predict.py
│   ├── cnn_predict.py
│   ├── lstm_predict.py
│   └── test_cnn_predictions.py
├── training/                  # Model training scripts
├── static/                    # Static frontend assets and uploads
├── templates/                 # Flask HTML templates
├── utils/                    # Utility modules
├── results/                   # Training results and reports
├── uploads/                  # Uploaded images for prediction
└── README.md
```

## Installation

1. Clone or download the project.
2. Open a terminal in the project root.
3. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

From the project root, start the Flask server:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Application routes

- `/` – landing page
- `/cnn` – chest X-ray prediction UI
- `/lstm` – vital-sign monitoring page
- `/anomaly` – patient anomaly detection page

## Model files

The app loads pre-trained models from the following locations:

- CNN: `models/cnn/deepmed_cnn_v2.keras`
- LSTM: `models/lstm/deepmed_lstm_v2.keras`
- Autoencoder: `models/autoencoder/deepmed_autoencoder_final.keras`

## Data and training

Training scripts are located in the `training/` folder. If you want to retrain or recalibrate the models, use those scripts as the starting point.

## Notes

- Uploaded images are stored in `static/uploads/`.
- The app runs in debug mode by default for local development.
- Model performance depends on the trained data and thresholds stored under `models/`.

## Disclaimer

This project is intended for educational, research, and prototype use. It is not a medical device and should not be used as a substitute for professional clinical assessment.
