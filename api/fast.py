import pandas as pd
from scam_job_detector.ML_logic.model import load_model
from scam_job_detector.ML_logic.data import clean_data
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
# import dill
from scam_job_detector.ML_logic.data import clean_data
from scam_job_detector.ML_logic.preprocessor import test_preprocessor
from scam_job_detector.ML_logic.model import load_model
from scam_job_detector.ML_logic.shapley import shapley


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        job_id: str = None,
        location: str = None,
        industry: str = None,
        function_str: str = None,
        employment_type: str = None,
        required_experience: str = None,
        required_education: str = None,
        has_company_logo: str = None,
        has_questions: str = None,
        department: str = None,
        salary_range: str = None,
        title: str = None,
        company_profile: str = None,
        description: str = None,
        requirements: str = None,
        telecommuting: str = None, 
        benefits: str = None,  # Predicted JD
    ):
    """
    Make a single course prediction.
    
    """
    # # Parse datetime string
    # eastern = pytz.timezone("US/Eastern")
    # pickup_dt_naive = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # pickup_dt_localized = eastern.localize(pickup_dt_naive)

    # # Convert to UTC if your model was trained on UTC timestamps
    # pickup_datetime_utc = pickup_dt_localized.astimezone(pytz.UTC)



    # Prepare the input DataFrame
    X_new = pd.DataFrame({
        'job_id': job_id,
        'location': location,
        'industry': industry,
        'function': function_str,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'has_company_logo': has_company_logo,
        'has_questions': has_questions,
        'department': department,
        'salary_range': salary_range,
        'title': title,
        'company_profile': company_profile,
        'description': description,
        'requirements': requirements,
        'telecommuting': telecommuting,
        'benefits': benefits
    }, index=[0])

    # Loads preprocessed features into X_processed variable
    X_new_cleaned = clean_data(X_new)
    X_new_preprocessed = test_preprocessor(X_new_cleaned)

    # Loading the model.
    model = load_model()

    # Generate prediction based up processed features.
    prediction = model.predict(X_new_preprocessed)[0]
    prediction_proba = model.predict_proba(X_new_preprocessed)[0][1].tolist()

    # shapley values
    shap_features_text, shap_text_values, shap_features_binary ,shap_values_binary, shap_features_country, shap_values_country = shapley(X_new_preprocessed)
    print(shap_features_text)


    return {
        "fraudulent": float(prediction),
        "prob_fraudulent": float(round(prediction_proba, 4)),
        'shap_features_text': shap_features_text,
        'shap_text_values': shap_text_values,
        'shap_features_binary': shap_features_binary,
        'shap_values_binary': shap_values_binary,
        'shap_features_country': shap_features_country,
        'shap_values_country': shap_values_country
    }


@app.get("/")
def root():
    return {"greeting": "Hello"}
