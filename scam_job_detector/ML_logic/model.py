import os
import dill
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, balanced_accuracy_score, f1_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scam_job_detector.ML_logic.preprocessor import train_preprocessor, test_preprocessor

def initialize_all_grid_searches(run_logreg=True, run_xgb=True):
    """
    Run grid-searches for baseline models only if requested.
    Preprocess data only once.
    Save best estimators for each model.
    Finally compute the best model ("winner model") based on test AP score.
    """

    # -----------------------------
    # Load cleaned dataset
    # -----------------------------
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")

    df = pd.read_csv(clean_data_path)
    print("✅ Clean data loaded")

    # -----------------------------
    # Train-test split
    # -----------------------------
    X = df.drop(columns=["fraudulent"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Preprocess once
    # -----------------------------
    X_train_pp = train_preprocessor(X_train)
    X_test_pp = test_preprocessor(X_test)

    # -----------------------------
    # Paths for saving models
    # -----------------------------
    models_folder = os.path.join(base_path, "models")
    os.makedirs(models_folder, exist_ok=True)

    logreg_path = os.path.join(models_folder, "model_logreg.dill")
    xgb_path = os.path.join(models_folder, "model_xgb.dill")
    winner_path = os.path.join(models_folder, "model_winner.dill")

    # Store results for winner selection
    model_scores = {}

    # ======================================================
    # 1️⃣ LOGISTIC REGRESSION GRID SEARCH (if requested)
    # ======================================================
    if run_logreg:
        print("\n🔍 Running Logistic Regression Grid Search...")

        param_grid_logreg = {
            'penalty': ['l1', 'l2'],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear']
        }

        grid_lr = GridSearchCV(
            LogisticRegression(),
            param_grid_logreg,
            cv=5,
            scoring='average_precision',
            n_jobs=-1
        )
        grid_lr.fit(X_train_pp, y_train)

        best_lr = grid_lr.best_estimator_

        with open(logreg_path, "wb") as f:
            dill.dump(best_lr, f)

        # Results
        print("✅ Grid search for LR completed")

        # Inspect best estimator:
        print(f"Best score: {grid_lr.best_score_}")
        print(f"Best parameters:, {grid_lr.best_params_}")
        print(f"Best estimator:, {grid_lr.best_estimator_}")

        # model performance on test set
        y_pred = grid_lr.predict(X_test_pp)
        print(f'''
            Model Performance
            Recall: {recall_score(y_test, y_pred)},
            Precision: {precision_score(y_test, y_pred)},
            Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
            F1 Score: {f1_score(y_test, y_pred)}
            ''')

    else:
        print("\n📂 Loading previously saved Logistic Regression model...")
        if os.path.exists(logreg_path):
            with open(logreg_path, "rb") as f:
                best_lr = dill.load(f)
        else:
            best_lr = None

    # Evaluate LR if available
    if best_lr is not None:
        y_pred_lr = best_lr.predict(X_test_pp)
        ap_lr = average_precision_score(y_test, y_pred_lr)
        model_scores["logreg"] = (ap_lr, best_lr)
        print(f"🔎 Logistic Regression AP on test: {ap_lr:.4f}")

    # ======================================================
    # 2️⃣ XGBOOST GRID SEARCH (if requested)
    # ======================================================
    if run_xgb:
        print("\n🔍 Running XGBoost Grid Search...")

        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, 5, 10]
        }

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )

        grid_xgb = GridSearchCV(
            xgb,
            param_grid_xgb,
            cv=5,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1
        )
        grid_xgb.fit(X_train_pp, y_train)

        best_xgb = grid_xgb.best_estimator_

        with open(xgb_path, "wb") as f:
            dill.dump(best_xgb, f)

        print(f"✅ Saved XGBoost model at {xgb_path}")
                # Results
        print("✅ Grid search for XGboost completed")

        # Inspect best estimator:
        print(f"Best score: {best_xgb.best_score_}")
        print(f"Best parameters:, {best_xgb.best_params_}")
        print(f"Best estimator:, {best_xgb.best_estimator_}")

        # model performance on test set
        y_pred = best_xgb.predict(X_test_pp)
        print(f'''
            Model Performance
            Recall: {recall_score(y_test, y_pred)},
            Precision: {precision_score(y_test, y_pred)},
            Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
            F1 Score: {f1_score(y_test, y_pred)}
            ''')


    else:
        print("\n📂 Loading previously saved XGBoost model...")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                best_xgb = dill.load(f)
        else:
            best_xgb = None

    # Evaluate XGB if available
    if best_xgb is not None:
        y_pred_xgb = best_xgb.predict(X_test_pp)
        ap_xgb = average_precision_score(y_test, y_pred_xgb)
        model_scores["xgb"] = (ap_xgb, best_xgb)
        print(f"🔎 XGBoost AP on test: {ap_xgb:.4f}")

    # ======================================================
    # 3️⃣ CHOOSE THE WINNER MODEL
    # ======================================================
    if len(model_scores) == 0:
        print("❌ No models available for comparison. Nothing to save.")
        return None

    winner_name = max(model_scores, key=lambda k: model_scores[k][0])
    winner_score, winner_model = model_scores[winner_name]

    with open(winner_path, "wb") as f:
        dill.dump(winner_model, f)

    print(f"\n🏆 WINNER MODEL: {winner_name}  (AP={winner_score:.4f})")
    print(f"💾 Saved at {winner_path}")

    return None

def load_model():
    """
    Load the model from the specified path
    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'model_winner.dill')

    with open(model_path, "rb") as file:
        model = dill.load(file)
    print("✅ Model loaded")
    return model

def load_preprocessor():
    """
    Load the preprocessor fitted with training data
    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')

    with open(model_path, "rb") as file:
        preprocessor = dill.load(file)
    print("✅ Preprocessor loaded")
    return preprocessor

if __name__ == "__main__":
    initialize_all_grid_searches()
    # load_model()




# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score
# # from scam_job_detector.ML_logic.data import clean_data
# from scam_job_detector.ML_logic.preprocessor import train_preprocessor, test_preprocessor
# from sklearn.model_selection import GridSearchCV
# import pandas as pd
# import os
# import dill

# def initialize_grid_search():
#     """
#     Initialize the Grid search for identifying the best model
#     Storing the best model
#     """
#     # Read file
#     clean_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
#     'raw_data',
#     'data_cleaned.csv'
#     )
#     print(clean_data_path)
#     df = pd.read_csv(clean_data_path)
#     print("✅ Clean data loaded")

#     # Extract X and y
#     X = df.drop(columns=['fraudulent'])
#     y = df['fraudulent']

#     # Make train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     # preprocess train and test data
#     X_train_preprocessed = train_preprocessor(X_train)
#     X_test_preprocessed = test_preprocessor(X_test)
    
#     # Defining parameters and scoring for grid search
#     param_grid = {
#         'penalty': ['l1', 'l2'],
#         'class_weight': [None, 'balanced'],
#         'solver': ['liblinear']
#     }


#     # Initializing grid search
#     grid_search = GridSearchCV(
#         LogisticRegression(),
#         param_grid,
#         cv=5,
#         scoring='average_precision',
#         n_jobs=-1
#     )

#     # Fit the grid search:
#     grid_search.fit(X_train_preprocessed, y_train)


#     # Results
#     print("✅ Grid search completed")

#     # Inspect best estimator:
#     print(f"Best score: {grid_search.best_score_}")
#     print(f"Best parameters:, {grid_search.best_params_}")
#     print(f"Best estimator:, {grid_search.best_estimator_}")

#     # model performance on test set
#     y_pred = grid_search.predict(X_test_preprocessed)
#     print(f'''
#           Model Performance
#           Recall: {recall_score(y_test, y_pred)},
#           Precision: {precision_score(y_test, y_pred)},
#           Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
#           F1 Score: {f1_score(y_test, y_pred)}
#         ''')

#     #timestamp = time.strftime("%Y%m%d-%H%M%S")
#     model = grid_search.best_estimator_
#     model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'model.dill')
#     #model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', f'model_{timestamp}.dill')

#     with open(model_path, "wb") as file:
#         dill.dump(model, file)
#     print(f"✅ Model saved at {model_path}")
#     return None

# # loading model

# def load_model():
#     """
#     Load the model from the specified path
#     """
#     model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'model.dill')

#     with open(model_path, "rb") as file:
#         model = dill.load(file)
#     print("✅ Model loaded")
#     return model

# def load_preprocessor():
#     """
#     Load the preprocessor fitted with training data
#     """
#     model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')

#     with open(model_path, "rb") as file:
#         preprocessor = dill.load(file)
#     print("✅ Preprocessor loaded")
#     return preprocessor

# if __name__ == "__main__":
#     # initialize_grid_search()
#     load_model()
