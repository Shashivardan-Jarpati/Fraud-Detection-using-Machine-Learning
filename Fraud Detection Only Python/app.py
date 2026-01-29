"""
Fraud Detection System using Machine Learning
==============================================

This Flask application provides a web interface for fraud detection in banking data
using three gradient boosting algorithms: CatBoost, XGBoost, and LightGBM.

Features:
---------
- CSV data upload and preprocessing
- Train-test split functionality
- Multiple ML model training (CatBoost, XGBoost, LightGBM)
- Model evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Fraud prediction on new data
- Interactive web interface

Author: Your Name
Date: 2024
Version: 1.0
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
import logging
from datetime import datetime
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for data storage
df = None
X_train, X_test, y_train, y_test = None, None, None, None
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Model instances
catboost_model = None
xgboost_model = None
lightgbm_model = None


class DataProcessor:
    """
    Handles all data preprocessing operations including loading, cleaning,
    and transformation of the dataset.
    """
    
    @staticmethod
    def validate_csv(file):
        """
        Validate uploaded CSV file.
        
        Args:
            file: FileStorage object from Flask request
            
        Returns:
            tuple: (bool, str) - (is_valid, message)
        """
        if not file:
            return False, "No file provided"
        
        if not file.filename.endswith('.csv'):
            return False, "Invalid file format. Please upload a CSV file."
        
        return True, "Valid file"
    
    @staticmethod
    def load_dataset(file):
        """
        Load CSV file into pandas DataFrame with error handling.
        
        Args:
            file: FileStorage object from Flask request
            
        Returns:
            tuple: (DataFrame or None, str) - (dataframe, message)
        """
        try:
            df = pd.read_csv(file)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            if df.empty:
                return None, "Uploaded file is empty"
            
            if df.shape[1] < 2:
                return None, "Dataset must have at least 2 columns (features and target)"
            
            return df, f"Dataset loaded successfully. {df.shape[0]} rows, {df.shape[1]} columns"
        
        except pd.errors.EmptyDataError:
            logger.error("Empty CSV file uploaded")
            return None, "The uploaded CSV file is empty"
        
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {str(e)}")
            return None, f"Error parsing CSV file: {str(e)}"
        
        except Exception as e:
            logger.error(f"Unexpected error loading dataset: {str(e)}")
            return None, f"Error loading dataset: {str(e)}"
    
    @staticmethod
    def preprocess_data(df):
        """
        Preprocess the dataset by handling missing values and data types.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            pandas DataFrame: preprocessed dataframe
        """
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            logger.warning("Missing values detected. Filling with appropriate values.")
            
            # Fill numeric columns with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    @staticmethod
    def get_dataset_info(df):
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        return info


class ModelTrainer:
    """
    Handles model training and evaluation for all three gradient boosting algorithms.
    """
    
    @staticmethod
    def train_catboost(X_train, y_train, X_test, y_test):
        """
        Train CatBoost classifier and return metrics.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            tuple: (model, metrics_dict)
        """
        logger.info("Training CatBoost model...")
        
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_state=42,
            train_dir="./catboost_info"
        )
        
        model.fit(X_train, y_train)
        metrics = ModelTrainer._calculate_metrics(model, X_test, y_test, "CatBoost")
        
        logger.info(f"CatBoost training completed. Accuracy: {metrics['accuracy']:.2f}%")
        return model, metrics
    
    @staticmethod
    def train_xgboost(X_train, y_train, X_test, y_test):
        """
        Train XGBoost classifier and return metrics.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            tuple: (model, metrics_dict)
        """
        logger.info("Training XGBoost model...")
        
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        metrics = ModelTrainer._calculate_metrics(model, X_test, y_test, "XGBoost")
        
        logger.info(f"XGBoost training completed. Accuracy: {metrics['accuracy']:.2f}%")
        return model, metrics
    
    @staticmethod
    def train_lightgbm(X_train, y_train, X_test, y_test):
        """
        Train LightGBM classifier and return metrics.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            tuple: (model, metrics_dict)
        """
        logger.info("Training LightGBM model...")
        
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        metrics = ModelTrainer._calculate_metrics(model, X_test, y_test, "LightGBM")
        
        logger.info(f"LightGBM training completed. Accuracy: {metrics['accuracy']:.2f}%")
        return model, metrics
    
    @staticmethod
    def _calculate_metrics(model, X_test, y_test, model_name):
        """
        Calculate comprehensive evaluation metrics for a trained model.
        
        Args:
            model: Trained classifier
            X_test, y_test: Testing data
            model_name: Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        y_pred = model.predict(X_test)
        y_probabilities = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred, average='weighted') * 100,
            'recall': recall_score(y_test, y_pred, average='weighted') * 100,
            'f1': f1_score(y_test, y_pred, average='weighted') * 100,
            'roc_auc': roc_auc_score(y_test, y_probabilities) * 100,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics


class PredictionService:
    """
    Handles fraud prediction on new data using trained models.
    """
    
    @staticmethod
    def predict_fraud(model, new_data):
        """
        Make fraud predictions on new data.
        
        Args:
            model: Trained classifier
            new_data: pandas DataFrame with features
            
        Returns:
            tuple: (predictions, probabilities)
        """
        try:
            predictions = model.predict(new_data)
            probabilities = model.predict_proba(new_data)[:, 1]
            
            # Map predictions to labels
            prediction_labels = ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions]
            
            logger.info(f"Predictions made for {len(predictions)} samples")
            return prediction_labels, probabilities
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise


# Flask route handlers
@app.route('/')
def index():
    """
    Render the main index page with dataset preview if available.
    """
    global df
    top_rows = None
    dataset_info = None
    
    if df is not None:
        top_rows = df.head().to_html(classes='table table-striped table-hover')
        dataset_info = DataProcessor.get_dataset_info(df)
    
    return render_template('index.html', df=df, top_rows=top_rows, dataset_info=dataset_info)


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle CSV file upload and display dataset preview.
    """
    global df
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return render_template('index.html', message='No file uploaded.')
    
    file = request.files['file']
    
    # Validate file
    is_valid, validation_message = DataProcessor.validate_csv(file)
    if not is_valid:
        return render_template('index.html', message=validation_message)
    
    # Load dataset
    df, message = DataProcessor.load_dataset(file)
    
    if df is not None:
        df = DataProcessor.preprocess_data(df)
        top_rows = df.head().to_html(classes='table table-striped table-hover')
        dataset_info = DataProcessor.get_dataset_info(df)
        return render_template('index.html', message=message, top_rows=top_rows, dataset_info=dataset_info)
    else:
        return render_template('index.html', message=message)


@app.route('/split', methods=['POST'])
def split():
    """
    Split the dataset into training and testing sets.
    """
    global df, X_train, X_test, y_train, y_test, label_encoder
    
    if df is None:
        return render_template('index.html', message='Please upload a dataset first.')
    
    try:
        # Separate features and target
        features = df.columns[:-1]
        X = df[features]
        y = df[df.columns[-1]]
        
        # Encode target variable
        y = label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Dataset split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        
        message = "Dataset split completed successfully."
        
        return render_template(
            'index.html',
            message=message,
            X_train_shape=X_train.shape,
            X_test_shape=X_test.shape,
            y_train_shape=y_train.shape,
            y_test_shape=y_test.shape
        )
    
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        return render_template('index.html', message=f'Error splitting dataset: {str(e)}')


@app.route('/run_catboost', methods=['POST'])
def run_catboost():
    """
    Train CatBoost model and display metrics.
    """
    global X_train, X_test, y_train, y_test, catboost_model
    
    if X_train is None or X_test is None:
        return render_template('index.html', message='Please upload and split the dataset first.')
    
    try:
        catboost_model, metrics = ModelTrainer.train_catboost(X_train, y_train, X_test, y_test)
        
        return render_template(
            'catboost_metrics.html',
            accuracy=round(metrics['accuracy'], 2),
            precision=round(metrics['precision'], 2),
            recall=round(metrics['recall'], 2),
            f1=round(metrics['f1'], 2),
            roc_auc=round(metrics['roc_auc'], 2)
        )
    
    except Exception as e:
        logger.error(f"CatBoost training error: {str(e)}")
        return render_template('index.html', message=f'Error training CatBoost: {str(e)}')


@app.route('/run_xgboost', methods=['POST'])
def run_xgboost():
    """
    Train XGBoost model and display metrics.
    """
    global X_train, X_test, y_train, y_test, xgboost_model
    
    if X_train is None or X_test is None:
        return render_template('index.html', message='Please upload and split the dataset first.')
    
    try:
        xgboost_model, metrics = ModelTrainer.train_xgboost(X_train, y_train, X_test, y_test)
        
        return render_template(
            'xgboost_metrics.html',
            accuracy=round(metrics['accuracy'], 2),
            precision=round(metrics['precision'], 2),
            recall=round(metrics['recall'], 2),
            f1=round(metrics['f1'], 2),
            roc_auc=round(metrics['roc_auc'], 2)
        )
    
    except Exception as e:
        logger.error(f"XGBoost training error: {str(e)}")
        return render_template('index.html', message=f'Error training XGBoost: {str(e)}')


@app.route('/run_lightgbm', methods=['POST'])
def run_lightgbm():
    """
    Train LightGBM model and display metrics.
    """
    global X_train, X_test, y_train, y_test, lightgbm_model
    
    if X_train is None or X_test is None:
        return render_template('index.html', message='Please upload and split the dataset first.')
    
    try:
        lightgbm_model, metrics = ModelTrainer.train_lightgbm(X_train, y_train, X_test, y_test)
        
        return render_template(
            'lightgbm_metrics.html',
            accuracy=round(metrics['accuracy'], 2),
            precision=round(metrics['precision'], 2),
            recall=round(metrics['recall'], 2),
            f1=round(metrics['f1'], 2),
            roc_auc=round(metrics['roc_auc'], 2)
        )
    
    except Exception as e:
        logger.error(f"LightGBM training error: {str(e)}")
        return render_template('index.html', message=f'Error training LightGBM: {str(e)}')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make fraud predictions on new data using the trained LightGBM model.
    """
    global lightgbm_model
    
    if lightgbm_model is None:
        return render_template('index.html', message='Please train the LightGBM model first.')
    
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded for prediction.')
    
    file = request.files['file']
    
    # Validate and load file
    is_valid, validation_message = DataProcessor.validate_csv(file)
    if not is_valid:
        return render_template('index.html', message=validation_message)
    
    try:
        new_data, message = DataProcessor.load_dataset(file)
        
        if new_data is None:
            return render_template('index.html', message=message)
        
        # Make predictions
        prediction_labels, probabilities = PredictionService.predict_fraud(lightgbm_model, new_data)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Prediction': prediction_labels,
            'Fraud Probability': [f"{prob:.2%}" for prob in probabilities]
        })
        
        predicted_table = results_df.to_html(classes='table table-striped table-hover')
        
        return render_template(
            'index.html',
            message='Prediction completed successfully.',
            predicted_table=predicted_table
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', message=f'Error making predictions: {str(e)}')


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return render_template('index.html', message='File too large. Maximum size is 16MB.'), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('index.html', message='An internal error occurred. Please try again.'), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('./catboost_info', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5100, debug=True)
