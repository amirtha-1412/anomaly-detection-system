# 🛡️ AI-Powered Network Anomaly Detection System

This project is an AI-based **Network Anomaly Detection System** that uses a Deep Learning model to detect malicious network activities in real-time. It integrates a **Flask backend** and a **React + Tailwind CSS frontend** to provide an interactive and user-friendly interface for anomaly detection and threat analysis.

---

## 🚀 Features

- 🔍 **Real-time anomaly detection**
- 🧠 **5-layer Deep Neural Network (DNN)**
- 📊 Predicts: DDoS, Port Scan, Brute Force, Probe attacks & more
- ⚡ **API Prediction Time:** ~1.3 seconds
- 🎯 **Model Accuracy:** 95.8%
- 🗂️ Uses **17 engineered network traffic features**
- 🖥️ Modern UI built with **React + Tailwind CSS**
- 🔗 Easy integration with other tools via REST API
- 🛑 Provides **risk levels**, **confidence scores**, and **threat indicators**
- 📁 Clean and modular folder structure

---

## 📁 Project Structure
ANOMALY-DETECTION-SYSTEM/
│
│── backend/
│ ├── app.py # Flask API
│ ├── anomaly_detection_model.py # Deep Learning model code
│ ├── anomaly_detection_model.h5 # Final trained model
│ ├── best_model.h5
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ ├── feature_cols.json
│ ├── model_metadata.json
│ ├── training_history.json
│ ├── sample_data.csv
│ └── requirements.txt # Python dependencies
│
│── frontend/
│ ├── public/
│ ├── src/
│ ├── package.json
│ ├── tailwind.config.js
│ └── vite.config.js
│
└── README.md

## ⚙️ Installation & Setup
cd backend
pip install -r requirements.txt

### Run the Flask server:
Your API will run at:
👉 **http://localhost:5000/predict**


## 🟩 2️⃣ Frontend Setup (React UI)
### Install dependencies:
cd frontend
npm install

### Start the development server:
npm start
Frontend will launch at:

👉 **http://localhost:3000/**

---

## 📈 Model Performance

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | **95.8%** |
| Precision     | **92.3%** |
| Recall        | **89.7%** |
| ROC-AUC       | **0.97**  |
| API Latency   | ~1.3 sec  |

---

## 📊 Technologies Used

### 🔹 Machine Learning & Backend
- Python  
- TensorFlow / Keras  
- Flask  
- Pandas, NumPy  
- Scikit-learn  
- KDD Cup 99 dataset  

### 🔹 Frontend
- React.js  
- Tailwind CSS  
- JavaScript  
- Vite  

---

## 🧠 How It Works (Simple Explanation)

1️⃣ Network features (duration, bytes, errors, failed logins, etc.) are input into the model.  
2️⃣ The DNN analyzes the data using learned patterns.  
3️⃣ The API returns:
   - Attack Type  
   - Confidence Score  
   - Risk Level  
   - Threat Indicators  
4️⃣ Frontend displays results with visual indicators and logs.

---

## 📦 Future Improvements

- Add live packet sniffing using Scapy  
- Deploy using Docker & Kubernetes  
- Add Graph Neural Network (GNN) model  
- Add database for storing logs  
- Real-time traffic streaming via Kafka  

---

## 🤝 Contributions

Pull requests are welcome!  
For suggestions, feel free to open an issue.