"""BERT embedding preprocessing utilities.

Provides functions to build embeddings from combined text columns and to fit/save
a lightweight non-text preprocessor for categorical & binary features.
Saved artifact: models/preprocessor_BERT.dill
"""

import os
import dill
import numpy as np
import pandas as pd
from pathlib import Path

# transformers imports are optional at import time to keep import cheap
from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


# Columns configuration — mirror the notebook
TEXT_FIELDS = [
    ("title", "TITLE"),
    ("description", "DESCRIPTION"),
    ("requirements", "REQUIREMENTS"),
    ("benefits", "BENEFITS"),
    ("company_profile", "COMPANY_PROFILE"),
]

# non-text columns
CATEGORICAL_COLUMNS = ["country", "industry", "employment_type"]
BINARY_COLUMNS = ["has_company_logo", "has_questions"]

# helper to join text columns into one field

def concat_job_row(row: pd.Series, sep: str = " [SEP] ") -> str:
    parts = []
    for col, tag in TEXT_FIELDS:
        txt = "" if row.get(col) is None else str(row.get(col))
        txt = " ".join(txt.split())
        if txt:
            parts.append(f"{tag}: {txt}")
    return sep.join(parts)


def bert_embed_meanpool(texts, model_name: str = "distilbert-base-uncased", max_length: int = 256, batch_size: int = 32, device: str | None = None):
    """Return mean-pooled BERT CLS embeddings for a list of texts.

    Args:
        texts (list[str])
        model_name (str): huggingface model name or path
        max_length (int)
        batch_size (int)
        device (str): 'cpu' or 'cuda' or None (auto-detect)

    Returns:
        np.ndarray shape (N, H)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        for k, v in enc.items():
            enc[k] = v.to(device)

        with torch.no_grad():
            out = model(**enc)
            hs = out.last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            summed = (hs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            meanpooled = summed / counts

        all_vecs.append(meanpooled.cpu().numpy())

    return np.vstack(all_vecs)


def _non_text_preprocessor_pipeline():
    """ColumnTransformer for categorical and binary columns (no text).

    The pipeline fits on raw X and can transform into a numpy array.
    """
    cat_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    binary_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent", fill_value=0),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = make_column_transformer(
        (cat_transformer, CATEGORICAL_COLUMNS),
        (binary_transformer, BINARY_COLUMNS),
        remainder="drop",
    )
    return preprocessor


def train_preprocessor_BERT(X_train: pd.DataFrame, model_name: str = "distilbert-base-uncased", max_length: int = 256, batch_size: int = 32, device: str | None = None):
    """Fit and persist the BERT preprocessor and compute transformed training features.

    Saves fitted non-text preprocessor and metadata at `models/preprocessor_BERT.dill`.

    Returns the combined numpy array of [bert_embeddings | non_text_transformed]
    """
    # ensure text field
    X = X_train.copy()
    if "full_text" not in X.columns:
        X["full_text"] = X.apply(concat_job_row, axis=1)

    # compute embeddings
    texts = X["full_text"].tolist()
    embeddings = bert_embed_meanpool(texts, model_name=model_name, max_length=max_length, batch_size=batch_size, device=device)

    # fit non-text preprocessor
    non_text_pre = _non_text_preprocessor_pipeline()
    non_text_fitted = non_text_pre.fit(X)
    non_text_transformed = non_text_fitted.transform(X)

    # persist artifact
    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "preprocessor_BERT.dill")
    artifact = {
        "non_text_preprocessor": non_text_fitted,
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
    }
    with open(preprocessor_path, "wb") as f:
        dill.dump(artifact, f)

    print(f"✅ Fitted BERT preprocessor saved at: {preprocessor_path}")

    # combine features
    if hasattr(non_text_transformed, "toarray"):
        non_text_arr = non_text_transformed.toarray()
    else:
        non_text_arr = non_text_transformed

    X_pp = np.hstack([embeddings, non_text_arr])
    return X_pp


def test_preprocessor_BERT(X_test: pd.DataFrame):
    """Load fitted artifact and transform new data into same feature space.

    Returns numpy array of shape (N, embedding_dim + non_text_dim)
    """
    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "preprocessor_BERT.dill")
    with open(preprocessor_path, "rb") as f:
        artifact = dill.load(f)

    non_text_fitted = artifact["non_text_preprocessor"]
    model_name = artifact["model_name"]
    max_length = artifact.get("max_length", 256)
    batch_size = artifact.get("batch_size", 32)

    X = X_test.copy()
    if "full_text" not in X.columns:
        X["full_text"] = X.apply(concat_job_row, axis=1)

    texts = X["full_text"].tolist()
    embeddings = bert_embed_meanpool(texts, model_name=model_name, max_length=max_length, batch_size=batch_size)

    non_text_transformed = non_text_fitted.transform(X)
    if hasattr(non_text_transformed, "toarray"):
        non_text_arr = non_text_transformed.toarray()
    else:
        non_text_arr = non_text_transformed

    X_pp = np.hstack([embeddings, non_text_arr])
    return X_pp


if __name__ == "__main__":
    # simple CLI to fit preprocessor on raw data
    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "raw_data" / "data_cleaned.csv"
    df = pd.read_csv(data_path)
    train_preprocessor_BERT(df)
    print("Done.")
