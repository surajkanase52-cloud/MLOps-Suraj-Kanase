from pathlib import Path
from ucimlrepo import fetch_ucirepo
from src.config import RAW


def download() -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=45)
    frame = dataset.data.features.copy()
    frame["num"] = dataset.data.targets.iloc[:, 0]
    frame.to_csv(RAW, index=False)
    return RAW


if __name__ == "__main__":
    print(download())
