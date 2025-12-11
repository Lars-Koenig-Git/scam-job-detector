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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
        job_id: str,
        location: str,
        industry: str,
        function_str: str,
        employment_type: str,
        required_experience: str,
        required_education: str,
        has_company_logo: str,
        has_questions: str,
        department: str,
        salary_range: str,
        title: str,
        company_profile: str,
        description: str,
        requirements: str,
        benefits: str,  # Predicted JD
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
        'benefits': benefits
    }, index=[0])

    # Loads preprocessed features into X_processed variable
    X_new_cleaned = clean_data(X_new)
    X_new_preprocessed = test_preprocessor(X_new_cleaned)

    # Loading the model.
    model = load_model()

    # Generate prediction based up processed features.
    prediction = model.predict(X_new_preprocessed)[0]

    return {
        "fraudulent": float(prediction),
    }


@app.get("/")
def root():
    return {"greeting": "Hello"}
