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


# catagorical columns for One-Hot Encoding
categorical_columns = [
    'country',
    'industry',
    'function',
    'employment_type'
]
# ordinal columns for Ordinal Encoding
ordinal_columns = [
    'required_experience',
    'required_education'
]
#binary columns for binary encoding
binary_columns = ['has_company_logo', 'has_questions', 'department_binary', 'salary_range_binary']

#text columns for TF-IDF Vectorizer
text_columns = [
        'title',
        'company_profile',
        'description',
        'requirements',
        'benefits'
]

#reference lists for ordinal encoding
experience_order = [
    "Not Applicable",
    "Unknown",
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive"
]

education_order = [
    "Unknown",
    "High School or equivalent",
    "Vocational",
    "Certification",
    "Some College Coursework Completed",
    "Associate Degree",
    "Bachelor's Degree",
    "Professional",
    "Master's Degree"
]


# preprocessor pipeline
def preprocessing_pipeline() -> ColumnTransformer:

    cat_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore')
    )
    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OrdinalEncoder(
        categories=[experience_order, education_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1)
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
    


    preprocessor = make_column_transformer(
        (cat_transformer, categorical_columns),
        (ordinal_transformer, ordinal_columns),
        (binary_transformer, binary_columns),
        (text_transformer, text_columns)
    )
    return preprocessor

# train preprocessor pipeline
def train_preprocessor(X_train: pd.DataFrame) -> np.ndarray:
    preprocessor = preprocessing_pipeline()
    X_train_fitted = preprocessor.fit(X_train)
    X_train_preprocessed = X_train_fitted.transform(X_train)

    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')
    with open(preprocessor_path, "wb") as file:
        dill.dump(X_train_fitted, file)

    print(f"Preprocessor saved at {preprocessor_path}")

    return X_train_preprocessed


# function to load and run the fitted preprocessor on new or test data
def test_preprocessor(X_test: pd.DataFrame) -> np.ndarray:
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')

    with open(model_path, "rb") as file:
        preprocessor = dill.load(file)

    X_test_preprocessed = preprocessor.transform(X_test)
    return X_test_preprocessed
