from flask import Flask, render_template, request , jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open(r"D:\Python-Course\ML-Course\test\Machine-Learning-Algorithm\K-Means\k_means_model.pkl", "rb"))

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        visit_score = float(request.form["visit_score"])
        spending_rank = float(request.form["spending_rank"])

        features = np.array([
            [visit_score, spending_rank]
        ])

        cluster = model.predict(features)[0]
        return render_template(
            "index.html",
            prediction_text=f"Assigned Cluster : {cluster}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)