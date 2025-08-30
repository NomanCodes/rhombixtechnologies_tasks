from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model.pkl not found. Run model.py to train the model first.")
        model = joblib.load(MODEL_PATH)
    return model

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pipe = get_model()

    # Accept JSON (AJAX) or form submission
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
    else:
        text = (request.form.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide non-empty text"}), 400

    label = pipe.predict([text])[0]
    confidence = None
    if hasattr(pipe[-1], "predict_proba"):
        try:
            confidence = float(pipe.predict_proba([text]).max(axis=1)[0])
        except Exception:
            confidence = None

    return jsonify({"label": str(label), "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
