from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

pipeline = pickle.load(open(
    r"D:\Python-Course\ML-Course\test\Machine-Learning-Algorithm\PCA\wine_pca_pipeline.pkl",
    "rb"
))

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [[
            float(request.form["fixed_acidity"]),
            float(request.form["volatile_acidity"]),
            float(request.form["citric_acid"]),
            float(request.form["residual_sugar"]),
            float(request.form["chlorides"]),
            float(request.form["free_sulfur_dioxide"]),
            float(request.form["total_sulfur_dioxide"]),
            float(request.form["density"]),
            float(request.form["pH"]),
            float(request.form["sulphates"]),
            float(request.form["alcohol"])
        ]]

        columns = [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]

        input_df = pd.DataFrame(values, columns=columns)

        prediction = pipeline.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Wine Quality: {prediction}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)