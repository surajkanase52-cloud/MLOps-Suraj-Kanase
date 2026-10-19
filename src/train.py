import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from src.config import PROCESSED, MODEL, METRICS, FIGURES

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def preprocessor():
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )


def candidates():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }


def main():
    df = pd.read_csv(PROCESSED)
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_experiment("heart-disease-classification")
    best = None
    summary = {}
    for name, model in candidates().items():
        pipe = Pipeline([("preprocess", preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=name):
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "precision", "recall", "roc_auc"],
            )
            pipe.fit(X_train, y_train)
            prob = pipe.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            metrics = {
                "cv_accuracy": scores["test_accuracy"].mean(),
                "cv_precision": scores["test_precision"].mean(),
                "cv_recall": scores["test_recall"].mean(),
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, pred),
                "test_precision": precision_score(y_test, pred, zero_division=0),
                "test_recall": recall_score(y_test, pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, prob),
            }
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model")
            FIGURES.mkdir(parents=True, exist_ok=True)
            RocCurveDisplay.from_predictions(y_test, prob)
            plt.tight_layout()
            roc = FIGURES / f"roc_{name}.png"
            plt.savefig(roc, dpi=160)
            plt.close()
            mlflow.log_artifact(str(roc))
            summary[name] = {k: float(v) for k, v in metrics.items()}
            if best is None or metrics["cv_roc_auc"] > best[0]:
                best = (metrics["cv_roc_auc"], name, pipe)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[2], MODEL)
    METRICS.write_text(
        json.dumps({"selected_model": best[1], "results": summary}, indent=2)
    )
    print(json.dumps({"selected_model": best[1], "results": summary}, indent=2))


if __name__ == "__main__":
    main()
