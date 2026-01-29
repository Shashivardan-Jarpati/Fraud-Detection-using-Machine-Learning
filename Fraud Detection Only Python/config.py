"""
Configuration Settings for Fraud Detection System
================================================

This module contains all configuration settings for the application,
including model parameters, file paths, and application settings.

Author: Your Name
Date: 2024
Version: 1.0
"""

import os
from pathlib import Path


class Config:
    """Base configuration class."""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Flask settings
    DEBUG = False
    TESTING = False
    HOST = '0.0.0.0'
    PORT = 5100
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/app.log'
    
    # Database settings (for future use)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///fraud_detection.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


class ModelConfig:
    """
    Configuration for machine learning models.
    Contains hyperparameters for all three models.
    """
    
    # CatBoost configuration
    CATBOOST_PARAMS = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'random_seed': 42,
        'verbose': False,
        'train_dir': './catboost_info',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'od_wait': 20
    }
    
    # XGBoost configuration
    XGBOOST_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic'
    }
    
    # LightGBM configuration
    LIGHTGBM_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': 42,
        'verbose': -1,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt'
    }
    
    # Training configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CROSS_VALIDATION_FOLDS = 5
    
    # Feature engineering
    SCALING_METHOD = 'standard'  # 'standard', 'minmax', or None
    HANDLE_IMBALANCE = False  # Whether to use SMOTE or other techniques
    FEATURE_SELECTION = False  # Whether to perform feature selection


class DataConfig:
    """Configuration for data processing."""
    
    # Data validation
    MIN_SAMPLES = 100
    MIN_FEATURES = 2
    MAX_MISSING_PERCENTAGE = 20  # Maximum percentage of missing values allowed
    
    # Data preprocessing
    FILL_MISSING_NUMERIC = 'median'  # 'mean', 'median', 'mode', or value
    FILL_MISSING_CATEGORICAL = 'mode'  # 'mode', 'unknown', or value
    
    # Outlier handling
    OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', or None
    OUTLIER_THRESHOLD_IQR = 1.5
    OUTLIER_THRESHOLD_ZSCORE = 3
    
    # Feature types
    EXCLUDE_FEATURES = []  # Features to exclude from training
    CATEGORICAL_FEATURES = []  # Explicitly specified categorical features
    NUMERICAL_FEATURES = []  # Explicitly specified numerical features


class PathConfig:
    """Configuration for file paths."""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Data directories
    DATA_DIR = BASE_DIR / 'data'
    UPLOAD_DIR = BASE_DIR / 'uploads'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    # Model directories
    MODEL_DIR = BASE_DIR / 'models'
    CATBOOST_DIR = BASE_DIR / 'catboost_info'
    
    # Log directories
    LOG_DIR = BASE_DIR / 'logs'
    
    # Report directories
    REPORT_DIR = BASE_DIR / 'reports'
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.UPLOAD_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.CATBOOST_DIR,
            cls.LOG_DIR,
            cls.REPORT_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class MetricsConfig:
    """Configuration for model evaluation metrics."""
    
    # Metrics to calculate
    METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'confusion_matrix',
        'classification_report'
    ]
    
    # Metric display precision
    DISPLAY_PRECISION = 2
    
    # Threshold for binary classification
    CLASSIFICATION_THRESHOLD = 0.5
    
    # Cross-validation settings
    CV_SCORING = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


class UIConfig:
    """Configuration for user interface elements."""
    
    # Color schemes for different models
    COLORS = {
        'catboost': '#FF6B6B',
        'xgboost': '#4ECDC4',
        'lightgbm': '#45B7D1'
    }
    
    # Chart settings
    FIGURE_DPI = 100
    FIGURE_FORMAT = 'png'
    
    # Table settings
    TABLE_CLASSES = 'table table-striped table-hover'
    MAX_ROWS_DISPLAY = 100


class SecurityConfig:
    """Security-related configuration."""
    
    # File upload security
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    # Session security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # CSRF protection
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None


class APIConfig:
    """Configuration for API endpoints (for future use)."""
    
    # API versioning
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'
    
    # Rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = '100 per hour'
    
    # API documentation
    API_TITLE = 'Fraud Detection API'
    API_DESCRIPTION = 'API for fraud detection using machine learning'


# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name='default'):
    """
    Get configuration object by name.
    
    Args:
        config_name: Name of configuration ('development', 'production', 'testing')
        
    Returns:
        Configuration class
    """
    return config_by_name.get(config_name, DevelopmentConfig)


def initialize_app_directories():
    """Initialize all required application directories."""
    PathConfig.create_directories()
    print("Application directories initialized successfully.")


if __name__ == "__main__":
    print("Configuration Module for Fraud Detection System")
    print("\nAvailable configurations:")
    for name in config_by_name.keys():
        print(f"  - {name}")
    
    print("\nInitializing directories...")
    initialize_app_directories()
    print("Done!")
