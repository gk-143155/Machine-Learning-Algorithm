from flask import Flask, render_template, request , jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open(r"D:\Python-Course\ML-Course\test\Machine-Learning-Algorithm\Random-Forest/random_forest_model.pkl", "rb"))

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        features = np.array(
            [[
                data["gender"],
                data["SeniorCitizen"],
                data["Partner"],
                data["Dependents"],
                data["tenure"],
                data["PhoneService"],
                data["MultipleLines"],
                data["InternetService"],
                data["OnlineSecurity"],
                data["OnlineBackup"],
                data["DeviceProtection"],
                data["TechSupport"],
                data["StreamingTV"],
                data["StreamingMovies"],
                data["Contract"],
                data["PaperlessBilling"],
                data["PaymentMethod"],
                data["MonthlyCharges"],
                data["TotalCharges"]
            ]]
        )

        prediction = model.predict(features)[0]
        show_message = "Customer Will Churn" if prediction == 1 else "Customer Will Stay"
        return render_template(
            "index.html",
            prediction_text=f" Prediction : {show_message}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)