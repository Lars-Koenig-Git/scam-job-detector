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

# catagorical columns for One-Hot Encoding
categorical_columns = [
    'country',
    'industry',
    'function',
    'employment_type',
]
# ordinal columns for Ordinal Encoding
ordinal_columns = [
    'required_experience',
    'required_education'
]
#binary columns for binary encoding
binary_columns = ['has_company_logo', 'has_questions']

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

#preprocessor pipeline

def feature_preprocessor(X: pd.DataFrame) -> np.ndarray:
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
        text_transformer = make_pipeline(
            TfidfVectorizer(max_features=5000)
        )
        preprocessor = make_column_transformer(
            (cat_transformer, categorical_columns),
            (ordinal_transformer, ordinal_columns),
            (binary_transformer, binary_columns),
            (text_transformer, text_columns)
        )
        return preprocessor

    preprocessor = preprocessing_pipeline()
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed
