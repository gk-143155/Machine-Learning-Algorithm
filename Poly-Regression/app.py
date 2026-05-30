from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open( r"D:\Python-Course\test-files\polynomial_regression_model.pkl" ,"rb"))

poly = pickle.load(open(r"D:\Python-Course\test-files\poly.pkl" , "rb"))

@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")
@app.route('/predict', methods=["POST"])
def predict():
    try:
            square_feet = int(request.form.get('square_feet', 0))
            bedrooms = int(request.form.get('bedrooms', 0))
            house_age = int(request.form.get('house_age', 0))
            Downtown = int(request.form['Downtown'])
            Suburban = int(request.form['Suburban'])
            TechPark = int(request.form['TechPark'])
        
            finput = np.array([[
                        square_feet,
                        bedrooms,
                        house_age,
                        Downtown,
                        Suburban,
                        TechPark
                        ]])
            x_poly = poly.transform(finput)
            prediction = model.predict(x_poly)[0]

            return render_template(
                "index.html",
                prediction_text=f"💰 Predicted Profit: {prediction:,.2f}"
            )

    except Exception as e:
            return render_template(
                "index.html",
                prediction_text=f"❌ Error: {str(e)}"
            )

if __name__ == "__main__":
    app.run(debug=True)
