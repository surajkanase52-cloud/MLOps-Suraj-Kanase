import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES, PROCESSED
import pandas as pd


def main():
    df = pd.read_csv(PROCESSED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df.hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.savefig(FIGURES / "histograms.png", dpi=160)
    plt.close()
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=160)
    plt.close()
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="target")
    plt.title("Class balance")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_balance.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
