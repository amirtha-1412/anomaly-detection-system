from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  

model = None
scaler = None
label_encoders = None
feature_cols = None
metadata = None

def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, label_encoders, feature_cols, metadata
    
    try:
        print("Loading model artifacts...")
        
        if os.path.exists('anomaly_detection_model.h5'):
            model = tf.keras.models.load_model('anomaly_detection_model.h5')
            print("✓ Model loaded")
        
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            print("✓ Scaler loaded")
        
        if os.path.exists('label_encoders.pkl'):
            label_encoders = joblib.load('label_encoders.pkl')
            print("✓ Label encoders loaded")
        
        if os.path.exists('feature_cols.json'):
            with open('feature_cols.json', 'r') as f:
                feature_cols = json.load(f)
            print("✓ Feature columns loaded")
        
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            print("✓ Metadata loaded")
        
        print("All artifacts loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

load_model_artifacts()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat(),
        'model_info': metadata if metadata else {}
    })

@app.route('/api/predict', methods=['POST'])
def predict_anomaly():
    """Predict if an access pattern is anomalous"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        data = request.json
        
        features = {
            'duration': float(data.get('duration', 0)),
            'src_bytes': float(data.get('src_bytes', 100)),
            'dst_bytes': float(data.get('dst_bytes', 100)),
            'wrong_fragment': int(data.get('wrong_fragment', 0)),
            'urgent': int(data.get('urgent', 0)),
            'hot': int(data.get('hot', 0)),
            'num_failed_logins': int(data.get('num_failed_logins', 0)),
            'logged_in': int(data.get('logged_in', 1)),
            'count': int(data.get('count', 10)),
            'srv_count': int(data.get('srv_count', 10)),
            'serror_rate': float(data.get('serror_rate', 0.0)),
            'srv_serror_rate': float(data.get('srv_serror_rate', 0.0)),
            'same_srv_rate': float(data.get('same_srv_rate', 1.0)),
            'diff_srv_rate': float(data.get('diff_srv_rate', 0.0))
        }
        
        if label_encoders:
            for cat_col in ['protocol_type', 'service', 'flag']:
                if cat_col in data and cat_col in label_encoders:
                    try:
                        encoded = label_encoders[cat_col].transform([data[cat_col]])[0]
                        features[cat_col + '_encoded'] = int(encoded)
                    except:
                        features[cat_col + '_encoded'] = 0
        
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        
        X_scaled = scaler.transform(X)
        prediction_proba = float(model.predict(X_scaled, verbose=0)[0][0])
        prediction = int(prediction_proba > 0.5)
        
        if prediction_proba < 0.3:
            risk_level = 'low'
        elif prediction_proba < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        threat_indicators = []
        if features['num_failed_logins'] > 2:
            threat_indicators.append('Multiple failed login attempts')
        if features['duration'] > 10000:
            threat_indicators.append('Unusually long connection duration')
        if features['serror_rate'] > 0.5:
            threat_indicators.append('High error rate detected')
        if features['wrong_fragment'] > 0:
            threat_indicators.append('Packet fragmentation anomaly')
        
        return jsonify({
            'success': True,
            'is_anomaly': bool(prediction),
            'confidence': round(prediction_proba * 100, 2),
            'risk_level': risk_level,
            'threat_indicators': threat_indicators,
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict anomalies for multiple records"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        data = request.json
        records = data.get('records', [])
        
        results = []
        for idx, record in enumerate(records):
            features = [record.get(col, 0) for col in feature_cols]
            X = np.array([features])
            X_scaled = scaler.transform(X)
            proba = float(model.predict(X_scaled, verbose=0)[0][0])
            
            results.append({
                'id': idx,
                'is_anomaly': bool(proba > 0.5),
                'confidence': round(proba * 100, 2),
                'risk_level': 'high' if proba > 0.7 else 'medium' if proba > 0.3 else 'low'
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics and dataset info"""
    try:
        stats = {
            'model_loaded': model is not None,
            'metadata': metadata if metadata else {},
            'timestamp': datetime.now().isoformat()
        }
        
        if os.path.exists('sample_data.csv'):
            df = pd.read_csv('sample_data.csv')
            if 'is_anomaly' in df.columns or 'label' in df.columns:
                if 'is_anomaly' not in df.columns:
                    df['is_anomaly'] = df['label'].apply(lambda x: 0 if 'normal' in str(x).lower() else 1)
                
                stats['dataset'] = {
                    'total_records': len(df),
                    'anomalies': int(df['is_anomaly'].sum()),
                    'normal': int((df['is_anomaly'] == 0).sum()),
                    'anomaly_rate': round(df['is_anomaly'].mean() * 100, 2)
                }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    try:
        info = {
            'model_loaded': model is not None,
            'architecture': {
                'layers': len(model.layers) if model else 0,
                'total_params': model.count_params() if model else 0
            },
            'features': {
                'count': len(feature_cols) if feature_cols else 0,
                'names': feature_cols if feature_cols else []
            },
            'metadata': metadata if metadata else {},
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'info': info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ANOMALY DETECTION API SERVER")
    print("="*60)
    print(f"Server starting on http://localhost:5000")
    print("Endpoints available:")
    print("  - GET  /api/health      : Health check")
    print("  - POST /api/predict     : Single prediction")
    print("  - POST /api/batch-predict : Batch predictions")
    print("  - GET  /api/stats       : Dataset statistics")
    print("  - GET  /api/model-info  : Model information")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)