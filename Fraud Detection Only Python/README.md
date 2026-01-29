# Fraud Detection System

A comprehensive machine learning application for detecting fraudulent transactions in banking data using gradient boosting algorithms.

## 🎯 Features

- **Multiple ML Algorithms**: CatBoost, XGBoost, and LightGBM
- **Interactive Web Interface**: Built with Flask and Bootstrap
- **Real-time Predictions**: Upload new data and get instant fraud predictions
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Data Visualization**: Visual representation of model performance
- **Easy to Use**: Simple upload, train, and predict workflow

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5100
```

## 📦 Dependencies

- Flask >= 2.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0
- xgboost >= 1.4.0
- catboost >= 0.26.0
- lightgbm >= 3.2.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 📊 Usage

### 1. Upload Dataset

- Click "Upload CSV Dataset"
- Select your fraud detection dataset
- The system will display the first 5 rows

### 2. Split Dataset

- Click "Split Dataset (Train/Test)"
- Data will be split into 80% training and 20% testing

### 3. Train Models

Choose one or more algorithms:
- **CatBoost**: Best for datasets with categorical features
- **XGBoost**: High performance and accuracy
- **LightGBM**: Fast training and prediction

### 4. Make Predictions

- Upload a new CSV file with the same structure
- Click "Predict Fraud"
- View the results with fraud probabilities

## 🏗️ Project Structure

```
fraud-detection/
│
├── app.py                  # Main Flask application
├── config.py              # Configuration settings
├── utils.py               # Utility functions
│
├── templates/             # HTML templates
│   ├── index.html
│   ├── catboost_metrics.html
│   ├── xgboost_metrics.html
│   └── lightgbm_metrics.html
│
├── static/                # Static files
│   └── styles.css
│
├── models/                # Saved models (created at runtime)
├── uploads/               # Uploaded datasets
├── logs/                  # Application logs
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

Edit `config.py` to customize:

- Model hyperparameters
- Training settings
- File upload limits
- Logging configuration

## 📈 Model Performance

The application calculates and displays:

- **Accuracy**: Overall correctness of the model
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🛠️ Advanced Features

### Data Preprocessing

- Automatic handling of missing values
- Outlier detection and treatment
- Feature scaling options
- Class imbalance handling

### Model Comparison

Compare multiple models side-by-side:
```python
from utils import ModelComparator

comparator = ModelComparator()
results = comparator.compare_models(model_metrics)
```

### Visualization

Generate charts and graphs:
```python
from utils import DataVisualizer

visualizer = DataVisualizer()
confusion_matrix_img = visualizer.plot_confusion_matrix(cm)
roc_curve_img = visualizer.plot_roc_curve(y_true, y_scores)
```

## 🔐 Security

- File upload size limits (16MB)
- Allowed file types: CSV only
- Input validation
- CSRF protection
- Session security

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- CatBoost team for the excellent gradient boosting library
- XGBoost developers for high-performance ML
- LightGBM team for fast and efficient boosting
- Flask community for the web framework
- Scikit-learn for machine learning utilities

## 📞 Support

For support, email your-email@example.com or open an issue in the repository.

## 🔮 Future Enhancements

- [ ] Add more ML algorithms (Random Forest, Neural Networks)
- [ ] Implement feature importance visualization
- [ ] Add model export/import functionality
- [ ] Create REST API endpoints
- [ ] Add user authentication
- [ ] Implement batch predictions
- [ ] Add model versioning
- [ ] Create interactive dashboards

## 📚 Documentation

For detailed documentation, visit [our wiki](https://github.com/yourusername/fraud-detection/wiki).

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

## 📊 Performance Benchmarks

Typical performance on standard hardware:

| Model | Training Time | Accuracy | Speed |
|-------|--------------|----------|-------|
| CatBoost | ~2-3 min | 95-98% | Fast |
| XGBoost | ~3-4 min | 94-97% | Medium |
| LightGBM | ~1-2 min | 95-98% | Very Fast |

## 🌟 Star History

If you find this project useful, please consider giving it a star!

---

Made with ❤️ for the open-source community
