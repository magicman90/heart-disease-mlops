
#This code will execute the data downloading form the given UCi Heart Disease dataset with 14 attributes
#Execution:
#python src/download_data.py

import io
import os
import ssl
import sys
import urllib.request

import certifi
import pandas as pd

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "diagnosis",
]


SOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "heart_disease_uci_raw.csv")

SSL_CONTEXTS = [
    ("certifi", ssl.create_default_context(cafile=certifi.where())),
    ("system default", ssl.create_default_context()),
]


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None
    for label, context in SSL_CONTEXTS:
        try:
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                return response.read()
        except Exception as e:
            last_error = e
            print(f"    [{label} SSL context failed: {e}]")
    raise last_error


def download() -> str:
    os.makedirs(RAW_DIR, exist_ok=True)

    for url in SOURCES:
        try:
            print(f"Attempting download from: {url}")
            raw_bytes = _fetch_bytes(url)

            if url.endswith(".data"):
                # Raw UCI file has no header, comma-separated, '?' for missing
                df = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=COLUMN_NAMES, na_values="?")
            else:
                df = pd.read_csv(io.BytesIO(raw_bytes), na_values="?")
                df.columns = [c.strip() for c in df.columns]

            df.to_csv(RAW_PATH, index=False)
            print(f"Success. Saved {df.shape[0]} rows x {df.shape[1]} cols to {RAW_PATH}")
            return RAW_PATH
        except Exception as e:
            print(f"  Failed ({e}). Trying next source...")

    print("ERROR: Could not download dataset from any source.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    download()
