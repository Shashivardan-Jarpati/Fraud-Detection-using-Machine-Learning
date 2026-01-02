# Fraud Detection in Banking Transactions
Flask-based web application for end-to-end fraud detection using gradient boosting models (CatBoost, XGBoost, LightGBM). Upload CSV data, preprocess, train models, evaluate metrics, and predict fraud via a responsive dashboard.
​

# 📋 Table of Contents 

Overview

Tech Stack

Key Features

Project Structure

Installation & Setup

Usage Workflow

Future Improvements

License

# Overview
This is a production-ready ML web app that processes banking transaction datasets to detect fraud. It handles the complete ML lifecycle:

Data Upload → Preview top rows

Preprocessing → Label encode target + 80/20 train/test split

Model Training → CatBoost/XGBoost/LightGBM with full metrics

Prediction → Batch fraud detection on new data

Sample Dataset: Bank-fraud-dataset.csv contains anonymized transactions with Time, V1-V28 (PCA features), Amount, and Class (0=Not Fraud, 1=Fraud).

# Tech Stack
** Backend & ML **
Flask (Web Framework)
Pandas (Data Processing)
scikit-learn (Preprocessing, Metrics, Splitting)
XGBoost (XGBClassifier)
CatBoost (CatBoostClassifier) 
LightGBM (LGBMClassifier)

** Frontend & UI **
HTML5 + Jinja2 Templating
Bootstrap 4 (Responsive Design)
Custom CSS (Animations, Gradients)
Google Fonts (Inter, Space Grotesk)

** Data & Templates **
Bank-fraud-dataset.csv (Labeled Training Data)
New-bank-dataset.csv (Prediction Input)
*_metrics.html (Model Performance Reports)

# Key Features

| Feature              | Description                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| CSV Upload & Preview | Secure file upload with top-5 row table preview ppl-ai-file-upload.s3.amazonaws​                         |
| Auto-Preprocessing   | LabelEncoder on target + 80/20 train/test split (random_state=21) ppl-ai-file-upload.s3.amazonaws​       |
| 3 Model Training     | CatBoost, XGBoost, LightGBM with accuracy/precision/recall/F1/ROC-AUC ppl-ai-file-upload.s3.amazonaws+2​ |
| Batch Prediction     | Upload new CSV → LightGBM predicts "Fraud"/"Not Fraud" ppl-ai-file-upload.s3.amazonaws​                  |
| Responsive UI        | Mobile-first Bootstrap cards, hover effects, error alerts ppl-ai-file-upload.s3.amazonaws​               |
| Session State        | Global variables persist dataset/models across routes ppl-ai-file-upload.s3.amazonaws​                   |

# Project Structure
fraud-detection-app/
├── app.py                          # Flask app + ML logic
├── templates/
│   ├── index.html                  # Main dashboard UI
│   ├── index1.html                 # Alternate template
│   ├── catboost_metrics.html       # CatBoost results
│   ├── xgboost_metrics.html        # XGBoost results
│   └── lightgbm_metrics.html       # LightGBM results
├── static/
│   └── index.css                   # Custom styling
├── Bank-fraud-dataset.csv          # Sample training data
├── New-bank-dataset.csv            # Sample prediction data

└── requirements.txt                # Dependencies

# Installation & Setup 
Prerequisites
-Python 3.8+
-pip
---Quick Start---
# 1. Clone repository
git clone <your-repo-url>
cd fraud-detection-app

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install flask pandas scikit-learn xgboost catboost lightgbm

# 4. Run application
python app.py
Open: http://127.0.0.1:5100

# Usage Workflow 
1. Upload Dataset
Dashboard → "Upload Dataset" card → Select CSV → Preview table
2. Preprocess & Split
"Data Preprocessing" card → "Split Dataset" → View train/test shapes
3. Train Models
Model Training section → Click "Run CatBoost" / "Run XGBoost" / "Run LightGBM"
→ Individual metrics page with scores
4. Make Predictions
"Make Predictions" → Upload new CSV → View "Fraud"/"Not Fraud" results table
# Error Handling: Clear messages guide users (e.g., "Upload dataset first").

# Future Improvements
 Model persistence (joblib/pickle)

 Feature importance plots

 Cross-validation

 Docker containerization

 REST API endpoints (FastAPI/Flask-RESTful)

 Real-time prediction (WebSocket)

# License
MIT License - Feel free to use, modify, and deploy commercially.



​

