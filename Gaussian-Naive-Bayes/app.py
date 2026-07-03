from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path("gaussian_bayes.pkl")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------

app = Flask(__name__)

logger.info("Loading trained model...")

model = joblib.load(MODEL_PATH)

logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def home():
    """
    Render home page.
    """

    return render_template(
        "index.html",
        features=FEATURES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict diabetes.
    """

    try:

        values = [
            float(request.form[feature])
            for feature in FEATURES
        ]

        input_df = pd.DataFrame(
            [values],
            columns=FEATURES,
        )

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Diabetic"
            color = "red"
        else:
            result = "Not Diabetic"
            color = "green"

        return render_template(
            "index.html",
            features=FEATURES,
            prediction=result,
            color=color,
        )

    except Exception:
        logger.exception("Prediction failed.")

        return render_template(
            "index.html",
            features=FEATURES,
            prediction="Something went wrong. Please check your inputs.",
            color="orange",
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)