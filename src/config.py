from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED = ROOT / "data" / "processed" / "heart.csv"
MODEL = ROOT / "artifacts" / "model.joblib"
METRICS = ROOT / "artifacts" / "metrics.json"
FIGURES = ROOT / "artifacts" / "figures"
COLUMNS = [
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
    "num",
]
FEATURES = COLUMNS[:-1]
