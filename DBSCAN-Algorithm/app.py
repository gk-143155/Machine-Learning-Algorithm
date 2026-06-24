from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open(r"D:\Python-Course\ML-Course\test\Machine-Learning-Algorithm\DBSCAN-Algorithm\dbscan_model.pkl", "rb"))

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        weight = float(request.form["weight"])
        height = float(request.form["height"])

        features = np.array([[weight, height]])

        # Scale input
        features_scaled = model["scaler"].transform(features)

        dbscan_model = model["dbscan_model"]
        labels = model["labels"]

        # Core points and their labels
        core_points = dbscan_model.components_
        core_labels = labels[dbscan_model.core_sample_indices_]

        distances = np.linalg.norm(
            core_points - features_scaled,
            axis=1
        )

        idx = np.argmin(distances)

        if distances[idx] <= dbscan_model.eps:
            cluster_value = int(core_labels[idx])
        else:
            cluster_value = -1

        return render_template(
            "index.html",
            prediction_text=f"Assigned Cluster : {cluster_value}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)