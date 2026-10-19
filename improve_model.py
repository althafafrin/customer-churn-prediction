"""Tune a Random Forest churn model using a validation set.

Run from the project folder:
    python improve_model.py

This uses separate train, validation, and test sets:
- training data fits the model
- validation data chooses a probability threshold
- test data estimates final performance only once
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

from train import load_and_clean, build_pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def metrics_at_threshold(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    y_binary = (y_true == "Yes").astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_binary, pred)),
        "precision_yes": float(precision_score(y_binary, pred, zero_division=0)),
        "recall_yes": float(recall_score(y_binary, pred, zero_division=0)),
        "f1_yes": float(f1_score(y_binary, pred, zero_division=0)),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put the Telco churn CSV in the project's data folder."
        )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 60% train, 20% validation, 20% final test; stratified at each split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    estimator = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    pipeline = build_pipeline(X_train, estimator)
    pipeline.fit(X_train, y_train)

    model_classes = list(pipeline.named_steps["model"].classes_)
    yes_idx = model_classes.index("Yes")

    val_prob = pipeline.predict_proba(X_val)[:, yes_idx]
    test_prob = pipeline.predict_proba(X_test)[:, yes_idx]

    # Select threshold on validation data only, maximizing churn-class F1.
    threshold_rows = [
        metrics_at_threshold(y_val, val_prob, t)
        for t in [round(v, 2) for v in
                  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]]
    ]
    threshold_df = pd.DataFrame(threshold_rows)
    best_row = threshold_df.sort_values(
        ["f1_yes", "recall_yes"], ascending=False
    ).iloc[0]
    chosen_threshold = float(best_row["threshold"])
    threshold_df.to_csv(REPORT_DIR / "threshold_results.csv", index=False)

    # Evaluate selected threshold on untouched test data.
    default_metrics = metrics_at_threshold(y_test, test_prob, 0.50)
    tuned_metrics = metrics_at_threshold(y_test, test_prob, chosen_threshold)
    tuned_pred = (test_prob >= chosen_threshold).astype(int)
    y_test_binary = (y_test == "Yes").astype(int)
    auc = float(roc_auc_score(y_test_binary, test_prob))

    results = {
        "selected_model": "Random Forest",
        "threshold_selected_on_validation": chosen_threshold,
        "validation_metrics_at_selected_threshold": best_row.to_dict(),
        "test_default_threshold_0_50": default_metrics,
        "test_tuned_threshold": tuned_metrics,
        "test_roc_auc": auc,
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test))
        },
        "note": "Threshold was chosen on validation data; final metrics use the separate test set."
    }

    with open(REPORT_DIR / "tuned_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save pipeline and the chosen threshold separately.
    joblib.dump(pipeline, MODEL_DIR / "churn_pipeline_tuned.joblib")
    with open(MODEL_DIR / "churn_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": chosen_threshold}, f, indent=2)

    cm = confusion_matrix(y_test_binary, tuned_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"], ax=ax
    )
    ax.set_title(f"Tuned Random Forest — threshold {chosen_threshold:.2f}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "tuned_confusion_matrix.png", dpi=180)
    plt.close(fig)

    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["preprocess"]
    importances = pd.DataFrame({
        "feature": prep.get_feature_names_out(),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv(REPORT_DIR / "tuned_feature_importance.csv", index=False)

    top = importances.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Tuned Random Forest — Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "tuned_feature_importance.png", dpi=180)
    plt.close(fig)

    print("\n=== Validation threshold comparison ===")
    print(threshold_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nChosen threshold: {chosen_threshold:.2f}")
    print("\n=== Test metrics: default threshold 0.50 ===")
    print(json.dumps(default_metrics, indent=2))
    print("\n=== Test metrics: tuned threshold ===")
    print(json.dumps(tuned_metrics, indent=2))
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print("\n=== Tuned test classification report ===")
    print(classification_report(
        y_test, ["Yes" if p else "No" for p in tuned_pred],
        labels=["No", "Yes"], zero_division=0
    ))
    print("\nSaved tuned model: models/churn_pipeline_tuned.joblib")
    print("Saved threshold: models/churn_threshold.json")
    print("Saved reports in reports/ (tuned_metrics.json, threshold_results.csv, "
          "tuned_confusion_matrix.png, tuned_feature_importance.png)")


if __name__ == "__main__":
    main()
