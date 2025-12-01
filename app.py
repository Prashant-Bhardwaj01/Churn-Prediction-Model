from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)

# load model files 
model = joblib.load("model/churn_model.joblib")
scaler = joblib.load("modle/scaler.joblib")
columns = joblib.load("model/columns.joblib")

@app.route("/")
def home():
    return render_template("index.html", columns = columns)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_data = []

        for col in columns:
            value = float(request.form[col])
            user_data.append(value)
        # Make DataFrame in correct order
        df = pd.DataFrame([user_data], columns = columns)

        # Scale the data
        scaled_data = scaler.transform(df)

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100

        result = "Customer Will CHURN" if prediction == 1 else "Customer Will STAY"

        return render_template("result.html",result=result, probability = round(probability,2))
    except Exception as e:
        return f"Error: {str(e)}"
if __name__ == "__main__":
    app.run(debug=True)