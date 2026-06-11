from flask import Flask, render_template, request # Flask is class # render_templete is function # requst is object created by flask automatically
import pickle
import numpy as np
app = Flask(__name__) # creating instance app from Flask class

# Load model
model = pickle.load(open(r"D:\Python-Course\ML-Course\test\Machine-Learning-Algorithm\Decision-Tree-Regression\ice_cream_sales_model.pkl", "rb"))


# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        weekend = int(request.form["weekend"])

        # Prepare input for model
        final_input = np.array([[
            temperature,
            humidity,
            weekend
        ]])

        # Prediction
        prediction = model.predict(final_input)[0]

        return render_template(
            "index.html",
            prediction_text=f"💰 Predicted Sales : {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)