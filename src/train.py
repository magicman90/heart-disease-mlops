
#This code is written to train the models Random FOrest and Logistic Regression on the heart disease dataset, 
#it stores every run to the MLflow with all expected details and saves the best model for reuse during inferencing using APIs
#Execution:
#python src/train.py


import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow 
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocessing import load_raw, clean_and_binarize, build_preprocessor, get_feature_target_split

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "heart_disease_uci_raw.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

RANDOM_STATE = 42

MODEL_GRID = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        },
    },
}


def evaluate(model, X_test, y_test, run_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {run_name}")
    cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{run_name}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}", color="#C44E52")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {run_name}")
    plt.legend()
    roc_path = os.path.join(FIG_DIR, f"roc_curve_{run_name}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return metrics, cm_path, roc_path


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment_name = "heart-disease-classification"
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    if existing is None:
        client.create_experiment(experiment_name, artifact_location=f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(experiment_name)

    raw = load_raw(RAW_PATH)
    df = clean_and_binarize(raw)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_overall = {"roc_auc": -1}

    for model_name, cfg in MODEL_GRID.items():
        with mlflow.start_run(run_name=model_name):
            preprocessor = build_preprocessor()
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", cfg["estimator"])])

            search = GridSearchCV(pipe, cfg["params"], cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_

            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("cv_folds", 5)

            metrics, cm_path, roc_path = evaluate(best_pipe, X_test, y_test, model_name)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_best_roc_auc", search.best_score_)

            mlflow.log_artifact(cm_path, artifact_path="plots")
            mlflow.log_artifact(roc_path, artifact_path="plots")

            mlflow.sklearn.log_model(best_pipe, artifact_path="model", serialization_format="pickle")

            print(f"\n[{model_name}] best params: {search.best_params_}")
            print(f"[{model_name}] test metrics: {metrics}")

            results.append({"model": model_name, **metrics, "best_params": search.best_params_})

            if metrics["roc_auc"] > best_overall["roc_auc"]:
                best_overall = {"model_name": model_name, "pipeline": best_pipe, **metrics}

    results_df = pd.DataFrame(results)
    results_csv = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print("\n=== Model comparison ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]])

    best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_overall["pipeline"], best_model_path)

    metadata = {
        "model_name": best_overall["model_name"],
        "metrics": {k: v for k, v in best_overall.items() if k not in ("model_name", "pipeline")},
        "feature_columns": list(X.columns),
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "model_metadata.joblib"))

    print(f"\nBest model: {best_overall['model_name']} (ROC-AUC={best_overall['roc_auc']:.4f})")
    print(f"Saved best model pipeline -> {best_model_path}")


if __name__ == "__main__":
    main()
