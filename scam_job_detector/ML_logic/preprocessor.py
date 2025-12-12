import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import FunctionTransformer
import os
import dill

# catagorical columns for One-Hot Encoding
categorical_columns = [
    'country',
    'industry',
    'function',
    'employment_type'
]

# binary columns for binary encoding
binary_columns = ['has_company_logo']

# text columns for TF-IDF Vectorizer
text_columns = [
        'title',
        'company_profile',
        'description',
        'requirements',
        'benefits'
]

# preprocessor pipeline
def preprocessing_pipeline(text=True) -> ColumnTransformer:

    cat_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore')
    )

    binary_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent', fill_value=0),
        OneHotEncoder(handle_unknown='ignore')
    )

    def combine_text(X):
        return X[text_columns].fillna("").agg(" ".join, axis=1)

    text_transformer = make_pipeline(
        FunctionTransformer(combine_text, validate=False),
        TfidfVectorizer(max_features=5000)
    )
    if text:
        preprocessor = make_column_transformer(
            (cat_transformer, categorical_columns),
            (binary_transformer, binary_columns),
            (text_transformer, text_columns)
        )
    else:
        preprocessor = make_column_transformer(
            (cat_transformer, categorical_columns),
            (binary_transformer, binary_columns),
            remainder='drop'
        )
    return preprocessor

# train preprocessor pipeline
def train_preprocessor(X_train: pd.DataFrame, text=True) -> np.ndarray:
    preprocessor = preprocessing_pipeline(text=text)
    X_train_fitted = preprocessor.fit(X_train)
    X_train_preprocessed = X_train_fitted.transform(X_train)

    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')
    with open(preprocessor_path, "wb") as file:
        dill.dump(X_train_fitted, file)

    print(f"Preprocessor saved at {preprocessor_path}")

    return X_train_preprocessed, preprocessor


# function to load and run the fitted preprocessor on new or test data
def test_preprocessor(X_test: pd.DataFrame) -> np.ndarray:
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')

    with open(model_path, "rb") as file:
        preprocessor = dill.load(file)

    X_test_preprocessed = preprocessor.transform(X_test)
    return X_test_preprocessed

if __name__ == "__main__":
    # initialize_grid_search()
    train_preprocessor()
