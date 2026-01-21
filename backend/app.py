from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for model and scalers
model = None
scaler_X = None
scaler_y = None

# -------------------------------
# LOAD MODEL AND SCALERS
# -------------------------------
def load_model_and_scalers():
    global model, scaler_X, scaler_y
    
    try:
        # Load the LSTM model
        model = load_model("fluoride_model.h5", compile=False)
        print("✓ LSTM Model loaded successfully")
        
        # Load the feature scaler (X)
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        print("✓ Feature scaler (X) loaded successfully")
        
        # Load the target scaler (y)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        print("✓ Target scaler (y) loaded successfully")
        
        print("\n" + "="*50)
        print("LSTM Fluoride Prediction Model Ready!")
        print("Model expects sequence input with window size = 10")
        print("="*50 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model or scalers: {e}")
        print("\nMake sure these files are in the same directory:")
        print("  - fluoride_model.h5")
        print("  - scaler_X.pkl")
        print("  - scaler_y.pkl")
        return False

# Load model on startup
load_model_and_scalers()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_fluoride(ph, ec, hardness, window=10):
    """
    Predict fluoride concentration using LSTM model
    The model expects a sequence of 10 timesteps
    For single prediction, we repeat the input 10 times
    """
    try:
        # Prepare input features
        X_new = np.array([[ph, ec, hardness]])
        
        # Scale the features
        X_new_scaled = scaler_X.transform(X_new)
        
        # Create sequence by repeating the input 10 times (window size)
        # Shape: (1, 10, 3) - 1 sample, 10 timesteps, 3 features
        X_seq = np.repeat(X_new_scaled.reshape(1, 1, 3), window, axis=1)
        
        # Make prediction
        pred_scaled = model.predict(X_seq, verbose=0)
        
        # Inverse transform to get actual fluoride value
        pred_fluoride = scaler_y.inverse_transform(pred_scaled)
        fluoride_value = float(pred_fluoride[0][0])
        
        # Ensure non-negative prediction
        fluoride_value = max(0, fluoride_value)
        
        return fluoride_value
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# -------------------------------
# API ENDPOINTS
# -------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fluoride concentration from water parameters
    Expected input: {'pH': float, 'ec': float, 'hardness': float}
    """
    try:
        # Check if model is loaded
        if model is None or scaler_X is None or scaler_y is None:
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Extract and validate features
        try:
            ph = float(data.get('pH', 7.0))
            ec = float(data.get('ec', 300))
            hardness = float(data.get('hardness', 150))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid input values. Please provide numeric values.'}), 400
        
        # Validate input ranges
        if not (0 <= ph <= 14):
            return jsonify({'error': 'pH must be between 0 and 14'}), 400
        if ec < 0:
            return jsonify({'error': 'EC must be positive'}), 400
        if hardness < 0:
            return jsonify({'error': 'Hardness must be positive'}), 400
        
        # Make prediction
        fluoride_value = predict_fluoride(ph, ec, hardness)
        
        # Determine risk level
        if fluoride_value < 1.0:
            risk_level = "Safe"
            risk_color = "#10b981"
        elif fluoride_value <= 1.5:
            risk_level = "Moderate"
            risk_color = "#fbbf24"
        else:
            risk_level = "Unsafe"
            risk_color = "#ef4444"
        
        # Prepare response
        response = {
            'fluoride': round(fluoride_value, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'input_parameters': {
                'pH': round(ph, 2),
                'ec': round(ec, 2),
                'hardness': round(hardness, 2)
            }
        }
        
        print(f"✓ Prediction: pH={ph:.2f}, EC={ec:.2f}, Hardness={hardness:.2f} → Fluoride={fluoride_value:.2f} mg/L ({risk_level})")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict fluoride for multiple samples
    Expected input: {'samples': [{'pH': float, 'ec': float, 'hardness': float}, ...]}
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        results = []
        
        for idx, sample in enumerate(samples):
            try:
                ph = float(sample.get('pH', 7.0))
                ec = float(sample.get('ec', 300))
                hardness = float(sample.get('hardness', 150))
                
                # Make prediction
                fluoride_value = predict_fluoride(ph, ec, hardness)
                
                # Determine risk level
                if fluoride_value < 1.0:
                    risk_level = "Safe"
                elif fluoride_value <= 1.5:
                    risk_level = "Moderate"
                else:
                    risk_level = "Unsafe"
                
                results.append({
                    'index': idx,
                    'input': {'pH': ph, 'ec': ec, 'hardness': hardness},
                    'fluoride': round(fluoride_value, 2),
                    'risk_level': risk_level
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({'predictions': results, 'count': len(results)}), 200
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get information about the model
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        info = {
            'model_type': 'LSTM Neural Network (Keras/TensorFlow)',
            'model_file': 'fluoride_model.h5',
            'architecture': 'LSTM with sequence window',
            'window_size': 10,
            'input_features': ['pH', 'EC (µS/cm)', 'Hardness (mg/L)'],
            'output': 'Fluoride concentration (mg/L)',
            'scalers': {
                'feature_scaler': 'scaler_X.pkl (MinMaxScaler)',
                'target_scaler': 'scaler_y.pkl (MinMaxScaler)'
            },
            'risk_thresholds': {
                'safe': '< 1.0 mg/L',
                'moderate': '1.0 - 1.5 mg/L',
                'unsafe': '> 1.5 mg/L'
            },
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape)
        }
        return jsonify(info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    try:
        if model is None or scaler_X is None or scaler_y is None:
            return jsonify({
                'status': 'unhealthy',
                'model_loaded': model is not None,
                'scalers_loaded': scaler_X is not None and scaler_y is not None
            }), 500
        
        # Quick test prediction
        test_fluoride = predict_fluoride(7.96, 395, 160)
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'scalers_loaded': True,
            'test_prediction': round(test_fluoride, 2)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    """
    Root endpoint with API documentation
    """
    return jsonify({
        'name': 'Fluoride Prediction API',
        'version': '1.0',
        'model': 'LSTM Neural Network',
        'endpoints': {
            'POST /predict': {
                'description': 'Predict fluoride for single sample',
                'input': {'pH': 'float', 'ec': 'float', 'hardness': 'float'},
                'example': {'pH': 7.96, 'ec': 395, 'hardness': 160}
            },
            'POST /batch-predict': {
                'description': 'Predict fluoride for multiple samples',
                'input': {'samples': [{'pH': 'float', 'ec': 'float', 'hardness': 'float'}]}
            },
            'GET /model-info': 'Get model information',
            'GET /health': 'Health check',
            'GET /': 'API documentation'
        }
    }), 200

# -------------------------------
# ERROR HANDLERS
# -------------------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# -------------------------------
# MAIN
# -------------------------------

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" 💧 FLUORIDE PREDICTION API - LSTM MODEL")
    print("="*60)
    print("\n🌐 Server starting on http://localhost:5000")
    print("\n📍 Available Endpoints:")
    print("   POST   http://localhost:5000/predict")
    print("   POST   http://localhost:5000/batch-predict")
    print("   GET    http://localhost:5000/model-info")
    print("   GET    http://localhost:5000/health")
    print("\n💡 Test with:")
    print('   curl -X POST http://localhost:5000/predict \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"pH": 7.96, "ec": 395, "hardness": 160}\'')
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')