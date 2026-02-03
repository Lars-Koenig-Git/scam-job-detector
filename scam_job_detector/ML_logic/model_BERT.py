"""BERT embeddings + XGBoost training utilities.

Designed to mirror the style of `model.py` but for a pipeline that converts
text to BERT mean-pooled embeddings and trains an XGBoost classifier on
[embeddings | non-text features].

Saved artifact: models/model_BERT.dill
"""

import os
import dill
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, balanced_accuracy_score, f1_score,
    average_precision_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from scam_job_detector.ML_logic.preprocess_BERT import train_preprocessor_BERT, test_preprocessor_BERT


def final_model_BERT():
    """Train final XGBoost on BERT embeddings and persist to disk."""
    # load cleaned dataset
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")
    df = pd.read_csv(clean_data_path)
    print("✅ Clean data loaded")

    # train-test split
    X = df.drop(columns=["fraudulent"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # preprocess (this also saves the preprocessor artifact)
    print("🔁 Preprocessing training data (BERT embeddings + non-text features)")
    X_train_pp = train_preprocessor_BERT(X_train)
    X_test_pp = test_preprocessor_BERT(X_test)

    # paths for saving models
    models_folder = os.path.join(base_path, "models")
    os.makedirs(models_folder, exist_ok=True)

    model_path = os.path.join(models_folder, "model_BERT.dill")

    # configure and train final xgboost model
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        learning_rate=0.05,
        min_child_weight=1,
        reg_lambda=1,
        max_depth=11,
        n_estimators=275,
    )

    print("🏋️ Training XGBoost on BERT features...")
    xgb.fit(X_train_pp, y_train)

    with open(model_path, "wb") as f:
        dill.dump(xgb, f)

    # evaluate on test set
    y_pred = xgb.predict(X_test_pp)
    y_pred_proba = xgb.predict_proba(X_test_pp)[:, 1]

    print(f"\n    Model Performance\n    Recall: {recall_score(y_test, y_pred)},\n    Precision: {precision_score(y_test, y_pred)},\n    Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},\n    F1 Score: {f1_score(y_test, y_pred)},\n    AUC: {roc_auc_score(y_test, y_pred_proba)}\n    ")

    print(f"✅ Saved BERT+XGB model at {model_path}")
    return None


def initialize_all_grid_searches_BERT(run_xgb: bool = True):
    """Optional grid search for XGBoost hyperparameters using BERT features."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")
    df = pd.read_csv(clean_data_path)

    X = df.drop(columns=["fraudulent"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # preprocess once
    X_train_pp = train_preprocessor_BERT(X_train)
    X_test_pp = test_preprocessor_BERT(X_test)

    models_folder = os.path.join(base_path, "models")
    os.makedirs(models_folder, exist_ok=True)

    if run_xgb:
        print("\n🔍 Running XGBoost Grid Search (this may take a while)...")
        param_grid_xgb = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
            "max_depth": [7, 11],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8],
            "reg_lambda": [1, 10],
        }

        xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=-1)

        grid_xgb = GridSearchCV(
            xgb,
            param_grid_xgb,
            cv=3,
            scoring="average_precision",
            n_jobs=2,
            verbose=1,
        )

        grid_xgb.fit(X_train_pp, y_train)

        best_xgb = grid_xgb.best_estimator_
        xgb_path = os.path.join(models_folder, "model_BERT_xgb_grid.dill")
        with open(xgb_path, "wb") as f:
            dill.dump(best_xgb, f)

        print(f"✅ Saved XGBoost grid-search model at {xgb_path}")
        print(f"Best score: {grid_xgb.best_score_}")

    print("Grid search completed.")
    return None


def load_model_BERT():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "model_BERT.dill")
    with open(model_path, "rb") as file:
        model = dill.load(file)
    print("✅ BERT+XGB model loaded")
    return model


def load_preprocessor_BERT():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "preprocessor_BERT.dill")
    with open(model_path, "rb") as file:
        pre = dill.load(file)
    print("✅ BERT preprocessor artifact loaded")
    return pre


if __name__ == "__main__":
    # simple behavior: train final model if not present
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_folder = os.path.join(base_path, "models")
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, "model_BERT.dill")

    if os.path.exists(model_path):
        load_model_BERT()
    else:
        final_model_BERT()
        load_model_BERT()
