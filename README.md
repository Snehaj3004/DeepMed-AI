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

The datasets used for training and testing are required for model training and experimentation.

### Chest X-Ray Dataset

The CNN model uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

**Dataset:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Download the dataset from Kaggle and extract it into the following location:

```text
datasets/
└── xray/
    └── chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── test/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── val/
            ├── NORMAL/
            └── PNEUMONIA/
```

The dataset contains `NORMAL` and `PNEUMONIA` chest X-ray images organized into training, testing, and validation sets.

### Patient Vitals Dataset

Place the patient vitals dataset at:

```text
datasets/
└── patient_vitals/
    └── patient_vitals.csv
```

The dataset is used by the LSTM model for patient vital-sign sequence analysis.

### Anomaly Detection Dataset

Place the synthetic health dataset at:

```text
datasets/
└── anomaly/
    └── synthetic_health_dataset_stratified_20_patients.csv
```

The dataset is used by the Autoencoder model for anomaly detection.

### Dataset Setup

1. Download the Chest X-Ray dataset from the Kaggle link above.
2. Extract the downloaded dataset.
3. Create the required `datasets/` folders in the project root.
4. Place each dataset in its corresponding folder.
5. Make sure the folder and file names match the structure shown above.
6. The training scripts in the `training/` folder can then be used to train or retrain the models.

The `datasets/` directory is excluded from Git tracking because of its size and is not included in this repository.


## Notes

- Uploaded images are stored in `static/uploads/`.
- The app runs in debug mode by default for local development.
- Model performance depends on the trained data and thresholds stored under `models/`.

## Disclaimer

This project is intended for educational, research, and prototype use. It is not a medical device and should not be used as a substitute for professional clinical assessment.
