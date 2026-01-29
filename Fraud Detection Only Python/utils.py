"""
Utility Functions for Fraud Detection System
===========================================

This module contains utility functions for data visualization,
model comparison, and report generation.

Author: Your Name
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io
import base64
from datetime import datetime
import json


class DataVisualizer:
    """
    Provides visualization functions for the fraud detection system.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm, title='Confusion Matrix'):
        """
        Create a confusion matrix visualization.
        
        Args:
            cm: Confusion matrix array
            title: Plot title
            
        Returns:
            str: Base64 encoded image
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return DataVisualizer._convert_plot_to_base64()
    
    @staticmethod
    def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
        """
        Create an ROC curve visualization.
        
        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            title: Plot title
            
        Returns:
            str: Base64 encoded image
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        return DataVisualizer._convert_plot_to_base64()
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=10, title='Feature Importance'):
        """
        Create a feature importance visualization.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            title: Plot title
            
        Returns:
            str: Base64 encoded image
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices], color='teal')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        return DataVisualizer._convert_plot_to_base64()
    
    @staticmethod
    def plot_model_comparison(metrics_dict, metric_name='accuracy'):
        """
        Create a bar chart comparing different models.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics as values
            metric_name: Name of the metric to compare
            
        Returns:
            str: Base64 encoded image
        """
        models = list(metrics_dict.keys())
        values = [metrics_dict[model][metric_name] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} Comparison Across Models')
        plt.ylim([0, 105])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return DataVisualizer._convert_plot_to_base64()
    
    @staticmethod
    def _convert_plot_to_base64():
        """
        Convert current matplotlib plot to base64 encoded string.
        
        Returns:
            str: Base64 encoded image
        """
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"


class ModelComparator:
    """
    Compare performance of different models.
    """
    
    @staticmethod
    def compare_models(model_metrics):
        """
        Compare multiple models and return comprehensive comparison.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            dict: Comparison results
        """
        comparison = {
            'best_accuracy': max(model_metrics.items(), 
                               key=lambda x: x[1]['accuracy'])[0],
            'best_precision': max(model_metrics.items(), 
                                key=lambda x: x[1]['precision'])[0],
            'best_recall': max(model_metrics.items(), 
                             key=lambda x: x[1]['recall'])[0],
            'best_f1': max(model_metrics.items(), 
                         key=lambda x: x[1]['f1'])[0],
            'best_roc_auc': max(model_metrics.items(), 
                              key=lambda x: x[1]['roc_auc'])[0],
            'overall_best': ModelComparator._calculate_overall_best(model_metrics)
        }
        
        return comparison
    
    @staticmethod
    def _calculate_overall_best(model_metrics):
        """
        Calculate overall best model based on average of all metrics.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            str: Name of best model
        """
        avg_scores = {}
        for model_name, metrics in model_metrics.items():
            avg_score = np.mean([
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['roc_auc']
            ])
            avg_scores[model_name] = avg_score
        
        return max(avg_scores.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def generate_comparison_table(model_metrics):
        """
        Generate a formatted comparison table.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            pandas.DataFrame: Comparison table
        """
        data = []
        for model_name, metrics in model_metrics.items():
            data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.2f}%",
                'Precision': f"{metrics['precision']:.2f}%",
                'Recall': f"{metrics['recall']:.2f}%",
                'F1 Score': f"{metrics['f1']:.2f}%",
                'ROC-AUC': f"{metrics['roc_auc']:.2f}%"
            })
        
        return pd.DataFrame(data)


class ReportGenerator:
    """
    Generate comprehensive reports for model performance.
    """
    
    @staticmethod
    def generate_html_report(model_metrics, dataset_info):
        """
        Generate an HTML report with all model performance metrics.
        
        Args:
            model_metrics: Dictionary of model metrics
            dataset_info: Information about the dataset
            
        Returns:
            str: HTML report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fraud Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric-box {{ background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Fraud Detection Model Performance Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            
            <h2>Dataset Information</h2>
            <div class="metric-box">
                <p><strong>Total Samples:</strong> {dataset_info.get('shape', ['N/A'])[0]}</p>
                <p><strong>Features:</strong> {dataset_info.get('shape', ['N/A', 'N/A'])[1]}</p>
            </div>
            
            <h2>Model Performance Comparison</h2>
        """
        
        # Add comparison table
        comparison_df = ModelComparator.generate_comparison_table(model_metrics)
        html += comparison_df.to_html(index=False, classes='table')
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    @staticmethod
    def save_report_to_file(report_content, filename):
        """
        Save report to a file.
        
        Args:
            report_content: Content to save
            filename: Output filename
            
        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False


class DataAnalyzer:
    """
    Provides data analysis functions.
    """
    
    @staticmethod
    def analyze_class_distribution(y):
        """
        Analyze class distribution in the target variable.
        
        Args:
            y: Target variable array/series
            
        Returns:
            dict: Distribution statistics
        """
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        total = len(y)
        percentages = {k: (v/total)*100 for k, v in distribution.counts()}
        
        return {
            'distribution': distribution,
            'percentages': percentages,
            'is_imbalanced': max(counts) / min(counts) > 2
        }
    
    @staticmethod
    def detect_outliers(df, method='iqr'):
        """
        Detect outliers in numeric columns.
        
        Args:
            df: pandas DataFrame
            method: Outlier detection method ('iqr' or 'zscore')
            
        Returns:
            dict: Outlier information for each column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:  # zscore
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > 3
            
            outliers[col] = {
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(df)) * 100
            }
        
        return outliers
    
    @staticmethod
    def generate_statistical_summary(df):
        """
        Generate comprehensive statistical summary.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Statistical summary
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'descriptive_stats': df[numeric_cols].describe().to_dict(),
            'correlations': df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {},
            'skewness': df[numeric_cols].skew().to_dict(),
            'kurtosis': df[numeric_cols].kurtosis().to_dict()
        }
        
        return summary


def format_metrics_for_display(metrics):
    """
    Format metrics dictionary for better display.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        dict: Formatted metrics
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = value
    return formatted


def validate_model_input(X):
    """
    Validate input data for model prediction.
    
    Args:
        X: Input features
        
    Returns:
        tuple: (is_valid, message)
    """
    if X is None or len(X) == 0:
        return False, "Input data is empty"
    
    if X.isnull().any().any():
        return False, "Input data contains missing values"
    
    if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return False, "All features must be numeric"
    
    return True, "Valid input"


def calculate_prediction_confidence(probabilities):
    """
    Calculate confidence levels for predictions.
    
    Args:
        probabilities: Array of prediction probabilities
        
    Returns:
        list: Confidence labels
    """
    confidences = []
    for prob in probabilities:
        if prob >= 0.9 or prob <= 0.1:
            confidences.append("High")
        elif prob >= 0.7 or prob <= 0.3:
            confidences.append("Medium")
        else:
            confidences.append("Low")
    return confidences


if __name__ == "__main__":
    print("Utility module for Fraud Detection System")
    print("This module provides helper functions and should be imported, not run directly.")
