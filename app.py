from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load trained model
model = joblib.load(BASE_DIR / "model.pkl")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True) or {}

    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "error": "News text is required."
        }), 400

    # Convert text into TF-IDF features
    features = vectorizer.transform([text])

    # Predict
    prediction = model.predict(features)[0]

    # Calculate confidence
    confidence = max(
        model.predict_proba(features)[0]
    ) * 100

    return jsonify({
        "label": str(prediction).upper(),
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
