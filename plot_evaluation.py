from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from train import load_and_clean

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline_tuned.joblib"
REPORT_DIR = ROOT / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Tuned model not found: {MODEL_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Recreate the 60/20/20 split used by improve_model.py.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)
    classes = list(model.named_steps["model"].classes_)
    yes_index = classes.index("Yes")
    churn_probabilities = probabilities[:, yes_index]
    y_test_binary = (y_test == "Yes").astype(int)

    # ROC curve
    auc = roc_auc_score(y_test_binary, churn_probabilities)
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test_binary,
        churn_probabilities,
        name="Random Forest",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("Customer Churn Prediction — ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    # Precision–Recall curve
    ap = average_precision_score(y_test_binary, churn_probabilities)
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        y_test_binary,
        churn_probabilities,
        name="Random Forest",
        ax=ax,
    )
    ax.set_title("Customer Churn Prediction — Precision–Recall Curve")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test Average Precision: {ap:.4f}")
    print(f"ROC curve saved to: {REPORT_DIR / 'roc_curve.png'}")
    print(
        "Precision–Recall curve saved to: "
        f"{REPORT_DIR / 'precision_recall_curve.png'}"
    )


if __name__ == "__main__":
    main()
