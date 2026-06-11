from flask import Flask, render_template, request # Flask is class # render_templete is function # requst is object created by flask automatically
import pickle
import numpy as np
app = Flask(__name__) # creating instance app from Flask class

# Load model
model = pickle.load(open(r"D:\Python-Course\ML-Course\test\Machine-Learning-Algorithm\Decision-Tree-Classifier\loan_approval_model.pkl", "rb"))


# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        income = int(request.form["income"])
        credit_score = int(request.form["credit_score"])

        # Prepare input for model
        final_input = np.array([[
            age,
            income,
            credit_score
        ]])

        # Prediction
        prediction = model.predict(final_input)[0]

        predictvalue = 'Approved' if prediction == 1 else 'Disapproved'

        return render_template(
            "index.html",
            prediction_text=f"💰 Loan Approval Status: {predictvalue}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)