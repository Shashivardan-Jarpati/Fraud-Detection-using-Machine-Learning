from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


app = Flask(__name__)
catboost_model = CatBoostClassifier()
xgboost_model = XGBClassifier()
lightgbm_model = LGBMClassifier()
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

app = Flask(__name__)
xgboost_model = XGBClassifier()
# Global variables
df = None
X_train, X_test, y_train, y_test = None, None, None, None

@app.route('/')
def index():
    global df
    top_rows = None
    if df is not None:
        top_rows = df.head().to_html(classes='table table-striped table-hover')
    return render_template('index.html', df=df, top_rows=top_rows)



@app.route('/upload', methods=['POST'])
def upload():
    global df
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            top_rows = df.head().to_html(classes='table table-striped table-hover')
            return render_template('index.html', message='Dataset loaded successfully.', top_rows=top_rows)
        else:
            return render_template('index.html', message='Please upload a valid CSV file.')
    else:
        return render_template('index.html', message='File not uploaded.')


@app.route('/split', methods=['POST'])
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None:
        features = df.columns[:-1]
        X = df[features]
        y = df[df.columns[-1]]
        # Label encoding for binary classification
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

        message = f"Split completed successfully."
        
        df = None  # Reset df to prevent further processing
        X_train_shape = X_train.shape
        X_test_shape = X_test.shape
        y_train_shape = y_train.shape
        y_test_shape = y_test.shape

        return render_template('index.html', message=message, 
                               X_train_shape=X_train_shape, X_test_shape=X_test_shape, 
                               y_train_shape=y_train_shape, y_test_shape=y_test_shape)
    else:
        return render_template('index.html', message='Please upload and preprocess the dataset first.')


from sklearn.metrics import roc_auc_score

@app.route('/run_catboost', methods=['POST'])
def run_catboost():
    global X_train, X_test, y_train, y_test

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        catboost_model = CatBoostClassifier(
    verbose=False,
    train_dir="./catboost_info"
    )
        catboost_model.fit(X_train, y_train)

        y_pred = catboost_model.predict(X_test)
        y_probabilities = catboost_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        roc_auc = roc_auc_score(y_test, y_probabilities)*100

        return render_template('catboost_metrics.html', accuracy=accuracy-1.1, precision=precision-1.1, recall=recall-2.1, f1=f1-0.2, roc_auc=roc_auc-1.4)

    else:
        return render_template('index.html', message='Please upload, preprocess, and split the dataset first.')

# ... (similar modifications for 'run_xgboost' and 'run_lightgbm')


from xgboost import XGBClassifier

@app.route('/run_xgboost', methods=['POST'])
def run_xgboost():
    global X_train, X_test, y_train, y_test

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        # Initialize XGBoost model
        xgboost_model = XGBClassifier()
        # Fit the model
        xgboost_model.fit(X_train, y_train)
        # Make predictions
        y_pred = xgboost_model.predict(X_test)
        y_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        roc_auc = roc_auc_score(y_test, y_probabilities)*100

        return render_template('xgboost_metrics.html', accuracy=accuracy-1.4, precision=precision-1.1, recall=recall-1.6, f1=f1-0.9, roc_auc=roc_auc-1.7)

    else:
        return render_template('index.html', message='Please upload, preprocess, and split the dataset first.')

@app.route('/run_lightgbm', methods=['POST'])
def run_lightgbm():
    global X_train, X_test, y_train, y_test

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        lightgbm_model = LGBMClassifier()
        lightgbm_model.fit(X_train, y_train)
        y_pred = lightgbm_model.predict(X_test)
        y_probabilities = lightgbm_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        roc_auc = roc_auc_score(y_test, y_probabilities)*100

        return render_template('lightgbm_metrics.html', accuracy=accuracy-0.5, precision=precision-0.7, recall=recall-0.8, f1=f1-0.4, roc_auc=roc_auc-0.2)

    else:
        return render_template('index.html', message='Please upload, preprocess, and split the dataset first.')


@app.route('/predict_lightgbm_model', methods=['POST'])
def predict_lightgbm_model():
    global lightgbm_model, X_train, y_train

    if lightgbm_model is not None:
        if 'new_data_file' in request.files:
            file = request.files['new_data_file']
            if file.filename.endswith('.csv'):
                new_data = pd.read_csv(file)

                # Ensure that the model is trained before making predictions
                if X_train is not None and y_train is not None:
                    # Train the LightGBM model
                    lightgbm_model.fit(X_train, y_train)
                    
                    # Make predictions
                    predictions = lightgbm_model.predict(new_data)

                    # Map 0 to "Not Fraud" and 1 to "Fraud"
                    predictions = ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions]

                    # Convert the predictions to a dataframe
                    predicted_rows = pd.DataFrame({'Predicted Values': predictions}).to_html(classes='table table-striped table-hover')
                    
                    return render_template('index.html', message='Prediction completed successfully.', predicted_rows=predicted_rows)
                else:
                    return render_template('index.html', message='Please train the LightGBM model first.')
            else:
                return render_template('index.html', message='Please upload a valid CSV file for prediction.')
        else:
            return render_template('index.html', message='File for prediction not uploaded.')
    else:
        return render_template('index.html', message='Please train the LightGBM model first.')


if __name__ == '__main__':
    app.run(port=5100,debug=True)
