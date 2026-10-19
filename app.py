"""Customer Churn Prediction — Streamlit application using the tuned model."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline_tuned.joblib"
THRESHOLD_PATH = ROOT / "models" / "churn_threshold.json"

NUMERIC_COLUMNS = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
ID_COLUMNS = {"customerid", "customer_id", "id"}
TARGET_COLUMNS = {"churn", "predictedchurn", "churnprobability"}


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("Machine learning demo for identifying customers who may leave a service.")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def load_threshold():
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            value = json.load(f).get("threshold", 0.40)
        value = float(value)
        return value if 0.0 < value < 1.0 else 0.40
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.40


def get_expected_columns(model):
    """Get raw input feature names from the fitted preprocessing step."""
    try:
        return list(model.named_steps["preprocess"].feature_names_in_)
    except (AttributeError, KeyError):
        return None


def clean_input_frame(df, model):
    """Remove ID/target fields, convert numeric fields, and align columns."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    drop_cols = [
        c for c in df.columns
        if c.lower() in ID_COLUMNS | TARGET_COLUMNS
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            # Telco data sometimes stores TotalCharges as text.
            df[col] = pd.to_numeric(df[col], errors="coerce")

    expected = get_expected_columns(model)
    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                "Your CSV/input is missing required feature columns: "
                + ", ".join(missing)
            )
        df = df[expected]

    return df


def predict_and_show(model, df, threshold):
    features = clean_input_frame(df, model)
    probabilities = model.predict_proba(features)
    classes = list(model.named_steps["model"].classes_)
    if "Yes" not in classes:
        raise ValueError("The saved model does not have a 'Yes' churn class.")

    churn_index = classes.index("Yes")
    churn_probs = probabilities[:, churn_index]
    labels = ["Yes" if p >= threshold else "No" for p in churn_probs]

    result = pd.DataFrame({
        "PredictedChurn": labels,
        "ChurnProbability": churn_probs
    })

    return result


# Load model only if the tuned model is present.
if not MODEL_PATH.exists():
    st.error(f"Tuned model not found: {MODEL_PATH}")
    st.info(
        "Run `python improve_model.py` from your project folder first. "
        "This app expects models/churn_pipeline_tuned.joblib."
    )
    st.stop()

try:
    pipeline = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the tuned model: {exc}")
    st.stop()

threshold = load_threshold()
st.sidebar.success(f"Using tuned model · threshold {threshold:.2f}")
mode = st.sidebar.radio(
    "Choose a workflow",
    ["Explore dataset", "Predict one customer", "Batch predictions"]
)


if mode == "Explore dataset":
    st.header("Explore churn data")
    uploaded = st.file_uploader("Upload the dataset CSV", type=["csv"], key="explore_csv")
    source = uploaded if uploaded is not None else (DATA_PATH if DATA_PATH.exists() else None)

    if source is None:
        st.info("Upload the Telco Customer Churn CSV to begin.")
    else:
        try:
            df = pd.read_csv(source)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total customers", f"{len(df):,}")
            c2.metric("Total columns", f"{df.shape[1]:,}")
            if "Churn" in df.columns:
                churn = df["Churn"].astype(str).str.strip().str.lower()
                c3.metric("Churn rate", f"{churn.isin(['yes', '1', 'true']).mean():.1%}")
                st.subheader("Dataset preview")
                st.dataframe(df.head(20), use_container_width=True)
                st.subheader("Churn distribution")
                st.bar_chart(df["Churn"].astype(str).value_counts())
            else:
                c3.metric("Churn rate", "N/A")
                st.warning("No 'Churn' target column was found in this CSV.")
                st.dataframe(df.head(20), use_container_width=True)
            st.subheader("Missing values by column")
            st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")


elif mode == "Predict one customer":
    st.header("Predict churn for one customer")
    st.write("Upload the training CSV to create fields with valid categories and numeric inputs.")
    uploaded = st.file_uploader("Upload the training CSV", type=["csv"], key="single_csv")

    source = uploaded if uploaded is not None else (DATA_PATH if DATA_PATH.exists() else None)
    if source is None:
        st.info("Upload WA_Fn-UseC_-Telco-Customer-Churn.csv to continue.")
    else:
        try:
            schema_df = pd.read_csv(source)
            schema_df.columns = schema_df.columns.astype(str).str.strip()
            expected = get_expected_columns(pipeline)
            feature_columns = expected or [
                c for c in schema_df.columns
                if c.lower() not in ID_COLUMNS | TARGET_COLUMNS
            ]

            left, right = st.columns(2)
            values = {}
            for i, col in enumerate(feature_columns):
                container = left if i % 2 == 0 else right
                series = schema_df[col] if col in schema_df.columns else pd.Series(dtype="object")

                with container:
                    if col in NUMERIC_COLUMNS:
                        numeric = pd.to_numeric(series, errors="coerce").dropna()
                        default = float(numeric.median()) if not numeric.empty else 0.0
                        if col == "SeniorCitizen":
                            values[col] = st.selectbox(col, [0, 1], index=int(default >= 0.5))
                        elif col == "tenure":
                            values[col] = st.number_input(
                                col, min_value=0, max_value=100,
                                value=max(0, min(100, int(round(default)))), step=1
                            )
                        else:
                            values[col] = st.number_input(
                                col, min_value=0.0, value=max(0.0, default),
                                step=0.01, format="%.2f"
                            )
                    else:
                        choices = sorted(series.dropna().astype(str).unique().tolist())
                        if not choices:
                            choices = ["Unknown"]
                        values[col] = st.selectbox(col, choices)

            st.divider()
            if st.button("Predict churn", type="primary"):
                try:
                    result = predict_and_show(pipeline, pd.DataFrame([values]), threshold)
                    probability = float(result.loc[0, "ChurnProbability"])
                    predicted = result.loc[0, "PredictedChurn"]

                    st.subheader("Prediction result")
                    if predicted == "Yes":
                        st.error("⚠️ Predicted outcome: Customer may churn")
                    else:
                        st.success("✅ Predicted outcome: Customer may stay")
                    st.metric("Estimated churn probability", f"{probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                    st.caption(
                        f"Classification threshold: {threshold:.2f}. "
                        "A probability at or above this threshold is classified as churn."
                    )
                    st.caption("This is a model estimate, not a guarantee of future behavior.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        except Exception as exc:
            st.error(f"Could not prepare input fields: {exc}")


else:
    st.header("Batch predictions")
    uploaded = st.file_uploader(
        "Upload customer records CSV (features only, or including Churn)",
        type=["csv"], key="batch_csv"
    )
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            st.subheader("Uploaded data preview")
            st.dataframe(batch.head(10), use_container_width=True)

            if st.button("Generate batch predictions", type="primary"):
                result = predict_and_show(pipeline, batch, threshold)
                output = batch.copy()
                output["PredictedChurn"] = result["PredictedChurn"].values
                output["ChurnProbability"] = result["ChurnProbability"].values
                st.subheader("Prediction results")
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


st.divider()
st.caption(
    "Educational demo. Predictions are estimates, not guarantees. "
    "Validate performance and thresholds before real-world use."
)
