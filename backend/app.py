from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# GLOBAL MODEL OBJECTS
# --------------------------------------------------
model = None
scaler_X = None
scaler_y = None

# --------------------------------------------------
# LOAD MODEL AND SCALERS
# --------------------------------------------------
def load_model_and_scalers():
    global model, scaler_X, scaler_y

    try:
        model = load_model(
            os.path.join(BASE_DIR, "fluoride_model.h5"),
            compile=False
        )
        print("✓ LSTM Model loaded successfully")

        with open(os.path.join(BASE_DIR, "scaler_X.pkl"), "rb") as f:
            scaler_X = pickle.load(f)
        print("✓ Feature scaler (X) loaded")

        with open(os.path.join(BASE_DIR, "scaler_y.pkl"), "rb") as f:
            scaler_y = pickle.load(f)
        print("✓ Target scaler (y) loaded")

        print("\n" + "="*50)
        print(" FLUORIDE PREDICTION MODEL READY ")
        print("="*50 + "\n")

        return True

    except Exception as e:
        print(f"❌ Error loading model/scalers: {e}")
        return False


# Load model on startup
load_model_and_scalers()

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
def predict_fluoride(ph, ec, hardness, window=10):
    try:
        X_new = np.array([[ph, ec, hardness]])
        X_scaled = scaler_X.transform(X_new)

        X_seq = np.repeat(X_scaled.reshape(1, 1, 3), window, axis=1)

        pred_scaled = model.predict(X_seq, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)

        return max(0.0, float(pred[0][0]))

    except Exception as e:
        raise Exception(f"Prediction error: {e}")


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Fluoride Prediction API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Single prediction",
            "POST /batch-predict": "Batch prediction",
            "GET /model-info": "Model information",
            "GET /health": "Health check"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        ph = float(data.get("pH", 7.0))
        ec = float(data.get("ec", 300))
        hardness = float(data.get("hardness", 150))

        if not (0 <= ph <= 14):
            return jsonify({"error": "pH must be between 0 and 14"}), 400
        if ec < 0 or hardness < 0:
            return jsonify({"error": "EC and Hardness must be positive"}), 400

        fluoride = predict_fluoride(ph, ec, hardness)

        if fluoride < 1.0:
            risk, color = "Safe", "#10b981"
        elif fluoride <= 1.5:
            risk, color = "Moderate", "#fbbf24"
        else:
            risk, color = "Unsafe", "#ef4444"

        print(f"✓ Prediction → Fluoride={fluoride:.2f} mg/L ({risk})")

        return jsonify({
            "fluoride": round(fluoride, 2),
            "risk_level": risk,
            "risk_color": color,
            "input": {
                "pH": ph,
                "ec": ec,
                "hardness": hardness
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    try:
        samples = request.get_json().get("samples", [])
        results = []

        for i, s in enumerate(samples):
            try:
                fluoride = predict_fluoride(
                    float(s["pH"]),
                    float(s["ec"]),
                    float(s["hardness"])
                )
                risk = "Safe" if fluoride < 1.0 else "Moderate" if fluoride <= 1.5 else "Unsafe"

                results.append({
                    "index": i,
                    "fluoride": round(fluoride, 2),
                    "risk_level": risk
                })

            except Exception as e:
                results.append({"index": i, "error": str(e)})

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "LSTM (TensorFlow/Keras)",
        "window_size": 10,
        "features": ["pH", "EC", "Hardness"],
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape)
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        test = predict_fluoride(7.0, 300, 150)
        return jsonify({
            "status": "healthy",
            "test_prediction": round(test, 2)
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# --------------------------------------------------
# ENTRY POINT (DO NOT USE DEBUG)
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
