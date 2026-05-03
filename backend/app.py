from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Allow requests from your website

# Load model and vectorizer once when server starts
print("Loading model and vectorizer...")
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
print("Ready! Model loaded with", len(model.classes_), "diseases.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "MedGuard AI API is running",
        "diseases": len(model.classes_),
        "endpoint": "POST /predict"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "Send JSON with 'symptoms' key"}), 400

        symptoms_text = data["symptoms"]  # e.g. "fever headache cough fatigue"
        if not symptoms_text.strip():
            return jsonify({"error": "Symptoms cannot be empty"}), 400

        # Vectorize and predict
        X = vectorizer.transform([symptoms_text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Get top 5 diseases with confidence scores
        top5_idx = np.argsort(probabilities)[::-1][:5]
        top5 = [
            {
                "disease": model.classes_[i],
                "confidence": round(float(probabilities[i]) * 100, 1)
            }
            for i in top5_idx
        ]

        return jsonify({
            "top_prediction": prediction,
            "confidence": round(float(probabilities[top5_idx[0]]) * 100, 1),
            "top5": top5
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
