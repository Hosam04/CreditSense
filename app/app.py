import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---- Paths (relative to this file, so it works from any working directory) ----
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.preprocessing import CreditPreprocessor  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# Artifacts written by notebooks/Modeling_and_Evaluation.ipynb
MODEL_FILE = MODELS_DIR / "xgb.json"
CALIBRATOR_FILE = MODELS_DIR / "calibrator.pkl"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.json"

st.set_page_config(page_title="CreditSense - Credit Risk Assessment", page_icon="💳", layout="wide")


# ---- Cached loaders ----
@st.cache_resource
def load_preprocessor():
    return CreditPreprocessor.load(PREPROCESSOR_FILE)


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_FILE)
    return model


@st.cache_resource
def load_calibrator():
    return joblib.load(CALIBRATOR_FILE)


@st.cache_data
def load_config():
    return json.loads(THRESHOLD_FILE.read_text())


def build_raw_row(v: dict) -> pd.DataFrame:
    """One-row DataFrame with the same raw columns as the Kaggle data."""
    return pd.DataFrame([v])


# ---- Header ----
st.title("💳 CreditSense: Credit Risk Assessment")
st.write(
    "Estimates the probability that a borrower will be **90+ days past due within "
    "the next two years**, based on the *Give Me Some Credit* dataset."
)

# ---- Load artifacts ----
try:
    pre = load_preprocessor()
    model = load_model()
    calibrator = load_calibrator()
    cfg = load_config()
except Exception as e:
    st.error(
        f"❌ Could not load model files from `{MODELS_DIR}`: {e}\n\n"
        "Run `notebooks/Modeling_and_Evaluation.ipynb` first to create "
        "`xgb.json`, `calibrator.pkl` and `threshold.json`."
    )
    st.stop()

# ---- Sidebar: decision rule ----
with st.sidebar:
    st.header("⚙️ Model")
    st.caption(f"{cfg['model']} ({cfg['n_trees']} trees)")
    rule = st.radio(
        "Decision threshold",
        ["Cost-based", "F1-based"],
        help="Cost-based: minimizes expected loss "
             f"(missed default costs {cfg['cost_fn']:g}x a wrongly flagged good customer). "
             "F1-based: balances precision and recall on defaulters.",
    )
    threshold = cfg["threshold_cost"] if rule == "Cost-based" else cfg["threshold_f1"]
    st.caption(f"Threshold: **{threshold:.3f}**")
    st.caption(
        "A borrower is flagged *High Risk* when the calibrated probability is at or "
        "above the threshold."
    )

# ---- Inputs ----
st.subheader("📊 Borrower details")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**👤 Personal information**")
    age = st.number_input(
        "Age (years)", min_value=18, max_value=110, value=45,
        help="The borrower's age in years. Must be 18 or older.",
    )
    income_known = st.checkbox("I know the monthly income", value=True)
    income = st.number_input(
        "Monthly income ($)", min_value=0.0, value=5400.0, step=100.0,
        disabled=not income_known,
        help="Total gross income per month, before taxes. Untick the box above if unknown.",
    )
    dependents_known = st.checkbox("I know the number of dependents", value=True)
    dependents = st.number_input(
        "Number of dependents (children / family members supported)",
        min_value=0, max_value=20, value=0, disabled=not dependents_known,
        help="People financially supported by the borrower, not including the borrower. "
             "Untick the box above if unknown.",
    )

with c2:
    st.markdown("**💳 Credit & debt**")
    utilization = st.number_input(
        "Credit card usage (% of limit used, as a decimal)",
        min_value=0.0, value=0.15, step=0.05,
        help="Total balance on credit cards and personal credit lines divided by their total "
             "limit. Example: 0.30 = using 30% of the limit; 1.20 = over the limit.",
    )
    debt_ratio = st.number_input(
        "Debt-to-income ratio (monthly debt payments ÷ monthly income)",
        min_value=0.0, value=0.37, step=0.05,
        help="Example: 0.40 means 40% of monthly income goes to debt payments "
             "(loans, mortgage, credit cards, alimony, etc.).",
    )
    open_lines = st.number_input(
        "Open loans and credit cards (total count)",
        min_value=0, max_value=60, value=8,
        help="Number of currently open loans (car, personal, etc.) plus credit lines and cards.",
    )
    real_estate = st.number_input(
        "Mortgage / real estate loans (count)",
        min_value=0, max_value=30, value=1,
        help="Number of mortgage and real-estate loans, including home equity lines of credit.",
    )

with c3:
    st.markdown("**⏰ Late payments (last 2 years)**")
    late_30_59 = st.number_input(
        "Times paid 30–59 days late",
        min_value=0, max_value=20, value=0,
        help="How many times a payment was 30 to 59 days overdue.",
    )
    late_60_89 = st.number_input(
        "Times paid 60–89 days late",
        min_value=0, max_value=20, value=0,
        help="How many times a payment was 60 to 89 days overdue.",
    )
    late_90 = st.number_input(
        "Times paid 90+ days late",
        min_value=0, max_value=20, value=0,
        help="How many times a payment was 90 days or more overdue (most serious).",
    )

# ---- Predict ----
if st.button("Evaluate Credit Risk", type="primary"):
    raw = build_raw_row({
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income if income_known else np.nan,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents if dependents_known else np.nan,
    })

    try:
        features = pre.transform(raw)[cfg["feature_columns"]]       # clean + impute + engineer
        score = model.predict_proba(features)[:, 1]                 # raw model score (no scaler)
        proba = float(np.clip(calibrator.predict(score)[0], 0.001, 0.99))  # calibrated P(default)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("🎯 Result")

    m1, m2 = st.columns(2)
    m1.metric("Probability of serious delinquency", f"{proba * 100:.1f}%")
    m2.metric("Decision threshold", f"{threshold * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= threshold:
        st.error("⚠️ **High Risk** — the probability is above the decision threshold.")
    else:
        st.success("✅ **Low Risk** — the probability is below the decision threshold.")

    st.caption(
        "The probability is calibrated: among borrowers with a similar score, about this share "
        "defaulted in held-out data. The base rate is ~6.6%, so 15–20% is already well above "
        "average. This is a demo model, not a lending decision."
    )

    with st.expander("See the features sent to the model"):
        st.dataframe(features.T.rename(columns={features.index[0]: "value"}))