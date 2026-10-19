import pandas as pd
from src.config import COLUMNS, RAW, PROCESSED


def load_and_clean(path=RAW):
    df = pd.read_csv(path, na_values="?")
    if not set(COLUMNS).issubset(df.columns):
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df = df[COLUMNS].copy()
    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["target"] = (df.pop("num") > 0).astype(int)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    return df
