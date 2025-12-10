import pandas as pd
from scam_job_detector.ML_logic.model import load_model
from scam_job_detector.ML_logic.data import clean_data
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        job_desctiption: str,  # Predicted JD
    ):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # # Parse datetime string
    # eastern = pytz.timezone("US/Eastern")
    # pickup_dt_naive = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # pickup_dt_localized = eastern.localize(pickup_dt_naive)

    # # Convert to UTC if your model was trained on UTC timestamps
    # pickup_datetime_utc = pickup_dt_localized.astimezone(pytz.UTC)

    # Prepare the input DataFrame
    X = pd.DataFrame([{
        "description": job_desctiption,

    }])

       # Loads preprocessed features into X_processed variable
    X_processed = clean_data(X)

    # Loading the model.
    model = load_model()

    # Generate prediction based up processed features.
    prediction = model.predict(X_processed)[0]

    return {
        "fraudulent": float(prediction),

    }


@app.get("/")
def root():
    return {"greeting": "Hello"}
