from flask import Flask, render_template, request
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model
model = load_model("food_safety_lstm_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LENGTH = 150

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    symptom = ""

    if request.method == "POST":
        symptom = request.form.get("symptom", "")
        text = clean_text(symptom)
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(
            seq,
            maxlen=MAX_LENGTH,
            padding='pre',
            truncating='pre'
        )
        pred = model.predict(pad, verbose=0)
        prob = float(pred[0][0])

        predicted_class = 1 if prob > 0.5 else 0
        prediction = label_encoder.inverse_transform([predicted_class])[0]

        if predicted_class == 1:
            confidence = round(prob * 100, 2)
        else:
            confidence = round((1.0 - prob) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        symptom=symptom
    )

@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json() or {}
        symptom = data.get("symptom", "")
        if not symptom:
            return {"error": "No symptom or complaint text provided"}, 400

        text = clean_text(symptom)
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(
            seq,
            maxlen=MAX_LENGTH,
            padding='pre',
            truncating='pre'
        )
        pred = model.predict(pad, verbose=0)
        prob = float(pred[0][0])

        predicted_class = 1 if prob > 0.5 else 0
        prediction = label_encoder.inverse_transform([predicted_class])[0]

        if predicted_class == 1:
            confidence = round(prob * 100, 2)
        else:
            confidence = round((1.0 - prob) * 100, 2)

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "raw_probability": prob
        }
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)