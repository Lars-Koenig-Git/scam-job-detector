# -----------------------------
# 1. Use the same Python version as your venv
# -----------------------------
FROM python:3.10.6

# -----------------------------
# 2. Set the working directory inside the container
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Copy project files into the container
# -----------------------------
COPY scam_job_detector scam_job_detector
COPY api api
COPY models models
COPY requirements_test.txt requirements.txt
COPY setup.py setup.py

# -----------------------------
# 4. Install dependencies
# -----------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('punkt_tab')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet')" ]
# -----------------------------
# 5. Launch API using uvicorn
# - host 0.0.0.0 allows receiving external requests
# - port is defined by environment variable $PORT
# - app path assumes: taxifare/main.py → app = FastAPI()
# -----------------------------

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
