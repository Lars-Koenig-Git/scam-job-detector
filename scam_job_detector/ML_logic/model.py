import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    # YOUR CODE HERE

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    # YOUR CODE HERE

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    # YOUR CODE HERE

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history



##############################################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression

df = clean_data(df)
# Extract X and y
X = df.drop(columns=['fraudulent'])
y = df['fraudulent']
# Make train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# preprocess train and test data
X_train_preprocessed, X_test_preprocessed = train_preprocessor(X_train, X_test)
# X_test_preprocessed = test_preprocessor(X_test)

cv = StratifiedKFold(n_splits=5)

pipe = Pipeline([
    # ("clean_preproc", clean_preproc),
    ("classifier", LogisticRegression(max_iter=1000))
])

cross_val_score(pipe, X_train_preprocessed, y_train, cv=cv).mean()

cross_model = cross_validate(pipe, X_train_preprocessed, y_train, cv=cv)
cross_model


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score

model = LogisticRegression(max_iter=1000)

result = model.fit(X_train_preprocessed, y_train)

result.score

y_pred = result.predict(X_test_preprocessed)

print(recall_score(y_test, y_pred), precision_score(y_test, y_pred), accuracy_score(y_test, y_pred))

#cross validation grid search

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    result,
    param_grid,
    cv=5,
    scoring="precision",
    n_jobs=-1
)
#Fit the grid search:
grid_search.fit(X_train_preprocessed, y_train)
#Inspect best estimator:
print(f"Best score: {grid_search.best_score_}")
print(f"Best parameters:, {grid_search.best_params_}")
print(f"Best estimator:, {grid_search.best_estimator_}")
