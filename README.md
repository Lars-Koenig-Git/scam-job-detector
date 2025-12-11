# 🕵️‍♀️ Fake Job Scam Detector
Identify fraudulent job postings using ML, NLP and deep learning.

---

# 📌 Description
This project aims to detect fraudulent job postings using:

- Logistic Regression + TF-IDF
- BERT embeddings (optional)
- XGBoost (optional)
- A full preprocessing pipeline (One-Hot + Ordinal + TF-IDF)
- FastAPI deployment
- Streamlit UI

The goal is to assist job seekers by flagging potentially fraudulent jobs before they waste time or expose sensitive data.

---

# 📂 Repository Structure
```bash
scam-job-detector/
│
├── api/
│   ├── fast.py                 # FastAPI application
│   └── __init__.py
│
├── build/                      # Auto-generated during packaging
│   └── lib/
│       └── scam_job_detector/
│           ├── ML_logic/
│           ├── params.py
│           └── utils.py
│
├── models/
│   ├── model.dill              # Main trained model (LogReg)
│   ├── model_logreg.dill       # Optional alternative model
│   ├── model_xgb.dill          # Optional XGBoost model
│   └── preprocessor.dill       # Saved preprocessing pipeline
│
├── notebooks/                  # All exploratory Jupyter notebooks
│   ├── data_inspection_syeda.ipynb
│   ├── FirstInspection_Lars.ipynb
│   └── gilles_eda.ipynb
│
├── raw_data/
│   ├── fake_job_postings.csv   # Original dataset
│   └── data_cleaned.csv        # Pre-cleaned dataset (optional)
│
├── scam_job_detector/
│   ├── ML_logic/
│   │   ├── data.py             # Text cleaning + feature engineering
│   │   ├── model.py            # GridSearch + training + saving
│   │   ├── preprocessor.py     # ColumnTransformer (OHE, Ordinal, TF-IDF)
│   │   └── __init__.py
│   │
│   ├── params.py               # Global parameters (if used)
│   ├── utils.py                # Utility functions
│   └── __init__.py
│
├── requirements.txt            # Package dependencies
├── requirements_dev.txt        # Dev dependencies (linting, formatting)
├── Dockerfile                  # Docker runtime definition
├── Makefile                    # CLI shortcuts for training, API, etc.
├── setup.py                    # Packaging config
├── README.md                   # Project documentation
└── tests/                      # Unit tests
```

---

# 🧹 Data Cleaning

Key preprocessing steps applied:

- Lowercasing all text
- Removing punctuation and numbers
- Removing English stopwords
- Lemmatization
- Filling missing text with `"missing value"`
- Creating binary indicators for missing important columns
- Extracting `country` from the `location` string
- Dropping irrelevant columns (`job_id`, `department`, `salary_range`, `location`)

The heavy text cleaning (tokenization, lemmatization, stopwords) is performed once in `data.py`.

---

# 🔧 Preprocessing Pipeline (sklearn)

Using a `ColumnTransformer` combining:

- **OneHotEncoder** → categorical columns
- **OrdinalEncoder** → ordered columns like experience & education
- **FunctionTransformer** → combines 5 text columns into one
- **TfidfVectorizer** → numeric vectors from text
- **SimpleImputer** → handles missing values safely

All preprocessing is fitted only on the **training split** to avoid data leakage.

---

# 🤖 Model Training

The baseline model uses **Logistic Regression** with:

- `solver="liblinear"`
- class imbalance handling (`class_weight="balanced"`)
- GridSearchCV with 5-fold stratification
- `average_precision` as the scoring metric

Metrics evaluated:

- Precision
- Recall
- F1 score
- Balanced accuracy

Final model is saved as:

- `models/model.dill` (trained classifier)
- `models/preprocessor.dill` (feature engineering pipeline)

The FastAPI service loads both artifacts during startup, applies the
preprocessor to incoming requests, and returns predictions from the model.

---

# 🔧 Setup Instructions

### Install Python environment
```bash
pyenv install 3.10.6
pyenv virtualenv 3.10.6 scam_job_detector
pyenv local scam_job_detector
pip install --upgrade pip
pip install -r requirements.txt

# Place dataset at:
# raw_data/fake_job_postings.csv

# Clean the dataset and generate data_cleaned.csv
python -m scam_job_detector.ML_logic.data

python -m scam_job_detector.ML_logic.model
# Outputs:
# models/model.dill
# models/preprocessor.dill

uvicorn api.fast:app --reload
# Visit:
# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs

streamlit run streamlit/app.py

---

# 🧪 Usage

### Load trained model
```python
from scam_job_detector.ML_logic.model import load_model
model = load_model()
