

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import json
import requests
from io import StringIO

def download_kdd_dataset():
    """Download KDD Cup 99 dataset (10% subset)"""
    print("Downloading KDD Cup 99 dataset...")
    
    url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    
    try:
        df = pd.read_csv(url, compression='gzip', names=columns)
        print(f"Dataset downloaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Using local backup or sample data...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create sample dataset if download fails"""
    print("Creating sample dataset...")
    np.random.seed(42)
    n = 5000
    
    data = {
        'duration': np.random.exponential(50, n),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh'], n),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], n),
        'src_bytes': np.random.exponential(1000, n),
        'dst_bytes': np.random.exponential(800, n),
        'wrong_fragment': np.random.randint(0, 3, n),
        'urgent': np.random.randint(0, 2, n),
        'hot': np.random.randint(0, 5, n),
        'num_failed_logins': np.random.randint(0, 5, n),
        'logged_in': np.random.randint(0, 2, n),
        'count': np.random.randint(1, 500, n),
        'srv_count': np.random.randint(1, 500, n),
        'serror_rate': np.random.uniform(0, 1, n),
        'srv_serror_rate': np.random.uniform(0, 1, n),
        'same_srv_rate': np.random.uniform(0, 1, n),
        'diff_srv_rate': np.random.uniform(0, 1, n),
        'label': np.random.choice(['normal', 'anomaly'], n, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

def preprocess_data(df, sample_size=50000):
    """Preprocess KDD dataset for anomaly detection"""
    print("\nPreprocessing data...")
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    df['is_anomaly'] = df['label'].apply(lambda x: 0 if x.strip() == 'normal.' else 1)
    
    print(f"Normal samples: {(df['is_anomaly'] == 0).sum()}")
    print(f"Anomaly samples: {(df['is_anomaly'] == 1).sum()}")
    
    feature_cols = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'same_srv_rate', 'diff_srv_rate'
    ]
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            feature_cols.append(col + '_encoded')
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].values
    y = df['is_anomaly'].values
    
    return X, y, feature_cols, label_encoders

def build_advanced_dnn(input_dim):
    """Build advanced Deep Neural Network with regularization"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def train_model():
    """Complete training pipeline"""
    
    df = download_kdd_dataset()
    
    X, y, feature_cols, label_encoders = preprocess_data(df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nBuilding Deep Neural Network...")
    model = build_advanced_dnn(X_train_scaled.shape[1])
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    
    print("\n" + "="*50)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*50)
    
    model.save('anomaly_detection_model.h5')
    print("✓ Model saved: anomaly_detection_model.h5")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Scaler saved: scaler.pkl")
    
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("✓ Label encoders saved: label_encoders.pkl")
    
    with open('feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    print("✓ Feature columns saved: feature_cols.json")
    
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f)
    print("✓ Training history saved: training_history.json")
    
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    sample_df.to_csv('sample_data.csv', index=False)
    print("✓ Sample data saved: sample_data.csv")
    
    metadata = {
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'anomaly_rate': float(y.mean()),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Metadata saved: model_metadata.json")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    return model, scaler, feature_cols, label_encoders

if __name__ == "__main__":
    print("="*50)
    print("ACCESS CONTROL ANOMALY DETECTION SYSTEM")
    print("Using KDD Cup 99 Dataset")
    print("="*50)
    train_model()