from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score
from scam_job_detector.ML_logic.data import clean_data
from scam_job_detector.ML_logic.preprocessor import train_preprocessor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import time
import os
import dill

def initialize_grid_search():
    """
    Initialize the Grid search for identifying the best model
    Storing the best model
    """
    # Read file
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'raw_data', 'fake_job_postings.csv')
    print(data_path)
    df = pd.read_csv(data_path)

    # clean data
    df = clean_data(df)

    # Extract X and y
    X = df.drop(columns=['fraudulent'])
    y = df['fraudulent']

    # Make train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # preprocess train and test data
    X_train_preprocessed, X_test_preprocessed = train_preprocessor(X_train, X_test)

    # Defining parameters and scoring for grid search
    param_grid = {
        'penalty': ['l1', 'l2', 'ElasticNet', 'None'],
        'class_weight': [None, 'balanced'],
        'solver': ['liblinear']
    }


    # Initializing grid search
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring='average_precision',
        n_jobs=-1
    )

    # Fit the grid search:
    grid_search.fit(X_train_preprocessed, y_train)


    # Results
    print("✅ Grid search completed")

    # Inspect best estimator:
    print(f"Best score: {grid_search.best_score_}")
    print(f"Best parameters:, {grid_search.best_params_}")
    print(f"Best estimator:, {grid_search.best_estimator_}")

    # model performance on test set
    y_pred = grid_search.predict(X_test_preprocessed)
    print(f'''
          Model Performance
          'Recall': {recall_score(y_test, y_pred)},
          'Precision': {precision_score(y_test, y_pred)},
          'Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
          'F1 Score: {f1_score(y_test, y_pred)}
        ''')

    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    model = grid_search.best_estimator_
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'model.dill')
    #model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', f'model_{timestamp}.dill')

    with open(model_path, "wb") as file:
        dill.dump(model, file)
    print(f"✅ Model saved at {model_path}")
    return None

# loading model

def load_model():
    """
    Load the model from the specified path
    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'model.dill')

    with open(model_path, "rb") as file:
        model = dill.load(file)
    print("✅ Model loaded")
    return model

if __name__ == "__main__":
    #initialize_grid_search()
    load_model()
