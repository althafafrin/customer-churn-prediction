"""Analyze the saved Customer Churn Prediction model.

Run from the project folder:
    python analyze_model.py
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train import load_and_clean

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
REPORT_DIR = ROOT / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the Telco churn CSV inside the project's data folder."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}\n"
            "Run `python train.py` first."
        )

    REPORT_DIR.mkdir(exist_ok=True)
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Match the same holdout split used by train.py.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)
    predictions = pipeline.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(
        y_test, predictions, labels=["No", "Yes"], zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed (No)", "Churned (Yes)"]
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Customer Churn — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix_detailed.png", dpi=180)
    plt.close(fig)

    # ROC-AUC from the saved pipeline, if probability scores are available.
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)
        classes = list(pipeline.named_steps["model"].classes_)
        if "Yes" in classes:
            auc = roc_auc_score(
                (y_test == "Yes").astype(int),
                probs[:, classes.index("Yes")]
            )
            print(f"ROC-AUC: {auc:.4f}")

    # Feature importance is available for tree-based models such as Random Forest.
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance.to_csv(
            REPORT_DIR / "feature_importance.csv", index=False
        )

        top = importance.head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["feature"], top["importance"])
        ax.set_title("Top 15 Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "feature_importance.png", dpi=180)
        plt.close(fig)

        print("\nTop 15 features:")
        print(importance.head(15).to_string(index=False))
        print(f"\nSaved feature chart: {REPORT_DIR / 'feature_importance.png'}")
        print(f"Saved feature table: {REPORT_DIR / 'feature_importance.csv'}")
    else:
        print(
            "\nThe saved model does not expose feature_importances_. "
            "If it is Logistic Regression, use coefficients instead."
        )

    print(f"\nSaved confusion matrix: {REPORT_DIR / 'confusion_matrix_detailed.png'}")


if __name__ == "__main__":
    main()
