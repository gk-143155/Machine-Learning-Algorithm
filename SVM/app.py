import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("svm_model.pkl")


# Feature names
FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:

        values = []

        for feature in FEATURES:
            values.append(float(request.form[feature]))

        df = pd.DataFrame([values], columns=FEATURES)

        prediction = model.predict(df)[0]

        if prediction == 1:
            result = "Malignant (Cancer Detected)"
            color = "red"
        else:
            result = "Benign (No Cancer)"
            color = "green"

        return render_template(
            "index.html",
            features=FEATURES,
            prediction=result,
            color=color,
        )

    except Exception as e:
        return render_template(
            "index.html",
            features=FEATURES,
            prediction=str(e),
            color="orange",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)