# CreditSense — Credit Risk Prediction

A complete machine learning system that estimates the probability a borrower will experience **serious financial distress (90+ days past due) within the next two years**. It covers the full pipeline: exploratory analysis, leakage-free preprocessing, model training with calibration, cost-aware threshold selection, and a ready-to-use Streamlit demo.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-red)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Notes and Limitations](#notes-and-limitations)
- [Acknowledgements](#acknowledgements)

---

## Overview

Banks use credit scoring to decide whether to approve a loan. **CreditSense** takes a borrower’s financial profile and returns a **calibrated probability of default** (`SeriousDlqin2yrs = 1`).

The main challenge is **class imbalance**: only ~**6.6%** of borrowers are positive cases. Accuracy alone is misleading, so the project optimizes for:

- **ROC-AUC** and **PR-AUC** for ranking quality
- **Isotonic calibration** so predicted probabilities match observed frequencies
- **Cost-aware threshold** (missed default costs 5× a false alarm) in addition to an F1-optimal threshold

The final model is a tuned **XGBoost** classifier (259 trees) with isotonic regression calibration.

---

## Dataset

- **Source:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)
- **Target:** `SeriousDlqin2yrs` — 1 if the person experienced 90+ days past due delinquency or worse within 2 years, otherwise 0
- **Size in this repo:**
  - `train.csv` — 104,805 labeled rows
  - `test.csv` — 45,195 unlabeled rows

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit-card / personal-line balance ÷ sum of credit limits |
| `age` | Age of the borrower (years) |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days past due (not worse) in the last 2 years |
| `DebtRatio` | Monthly debt payments ÷ monthly gross income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Number of mortgage / real-estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days past due (not worse) in the last 2 years |
| `NumberOfDependents` | Number of dependents (excluding the borrower) |

> The raw data is not redistributed under a separate license here. Download it from Kaggle and review the competition terms before any reuse.

---

## Project Structure

```
CreditSense/
├── app/
│   └── app.py                      # Streamlit interactive demo
├── data/
│   ├── train.csv                   # Raw labeled data
│   ├── test.csv                    # Raw unlabeled data
│   ├── X_train_final.csv           # Processed training features
│   ├── X_test_final.csv            # Processed hold-out features
│   ├── y_train_raw.csv             # Training labels
│   ├── y_test_raw.csv              # Hold-out labels
│   └── xgb_submission.csv          # Predicted probabilities for test.csv
├── models/
│   ├── xgb.json                    # Final XGBoost model (native format)
│   ├── calibrator.pkl              # IsotonicRegression calibrator
│   ├── preprocessor.json           # Training medians + feature order
│   └── threshold.json              # Thresholds + cost parameters
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb # Exploration + preprocessing
│   ├── Modeling_and_Evaluation.ipynb # Training, calibration, evaluation
│   └── preprocessing.py            # Shared CreditPreprocessor class
├── test/
│   └── test_model.ipynb            # Inference on the unlabeled test set
├── requirements.txt
├── .devcontainer/                  # VS Code / Codespaces setup (auto-starts Streamlit)
└── README.md
```

---

## Methodology

### 1. Data splitting
`train.csv` is split **80 / 20** into train and hold-out with `train_test_split(..., random_state=42, stratify=y)`. Both splits keep the same positive rate (**6.63%**).

### 2. Preprocessing (`notebooks/preprocessing.py`)
All cleaning and feature engineering live in a single `CreditPreprocessor` class that is **fitted on the training split only** and reused for evaluation and inference. This prevents train/serve skew and data leakage.

- **Missing values:** `MonthlyIncome` (~20% missing) and `NumberOfDependents` receive binary `…Missing` indicator columns, then are imputed with the **training median**.
- **Invalid sentinel codes:** values `96` and `98` in the three past-due columns are treated as placeholders → flagged (`…InvalidFlag`) and replaced by NaN before imputation.
- **Invalid ages:** ages below 18 are set to missing.
- **Persistence:** learned medians and final column order are saved to `models/preprocessor.json` (JSON, no custom-class pickling needed).

### 3. Feature engineering

| Engineered feature | Formula |
|---|---|
| `TotalMonthlyDebt` | `MonthlyIncome × DebtRatio` |
| `IncomePerDependentAdjusted` | `MonthlyIncome / (NumberOfDependents + 1)` |
| `TotalDelinquencies` | Sum of the three past-due counters |
| `RealEstateToTotalCredit` | `NumberRealEstateLoansOrLines / (NumberOfOpenCreditLinesAndLoans + 1)` |

`TotalDelinquencies` is the feature most strongly correlated with the target.

### 4. Modeling
- **Algorithm:** XGBoost with early stopping (final model uses **259 trees**)
- **Hyper-parameters (selected):** `learning_rate=0.05`, `max_depth=3`, `min_child_weight=10`, `subsample=0.8`, monotonic constraints on delinquency-related features
- **Cross-validation:** 5-fold stratified CV → **CV AUC = 0.864 ± 0.003**
- **Calibration:** Isotonic regression fitted on out-of-fold predictions so that predicted probabilities match observed default rates

### 5. Threshold selection
Two decision thresholds are computed on the calibrated out-of-fold scores:

| Rule | Threshold | Criterion |
|---|---|---|
| **Cost-based** (default) | ≈ 0.162 | Minimizes expected cost (FN cost = 5, FP cost = 1) |
| **F1-based** | ≈ 0.231 | Maximizes F1 on the positive class |

Both are stored in `models/threshold.json` and selectable in the Streamlit app.

---

## Results

Evaluated on the **20,961-row hold-out set** (1,390 positives) after calibration:

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.866** |
| **PR-AUC** | **0.403** |
| Brier score (calibrated) | 0.049 |
| Mean predicted vs actual rate | 0.066 vs 0.066 |

**At the cost-optimal threshold (0.162):**

| Metric | Value |
|---|---|
| Accuracy | 0.903 |
| Precision (positive) | 0.351 |
| Recall (positive) | 0.549 |
| F1 (positive) | 0.428 |
| NPV | 0.967 |

**At the F1-optimal threshold (0.231):**

| Metric | Value |
|---|---|
| Accuracy | 0.922 |
| Precision (positive) | 0.419 |
| Recall (positive) | 0.462 |
| F1 (positive) | 0.440 |

**Lift (risk ranking):**
- Top 5% riskiest borrowers capture **36.5%** of all defaulters
- Top 10% capture **54.0%**
- Top 20% capture **72.2%**

---

## Getting Started

### Prerequisites
- Python **3.11**
- (Optional) Jupyter for the notebooks

### Installation

```bash
# 1. Clone / extract the project
cd CreditSense

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Data
The repository already includes the processed data files and model artifacts.  
If you want to re-run the full pipeline from scratch, download `cs-training.csv` / `cs-test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them in `data/` as `train.csv` and `test.csv`.

---

## Usage

Run the notebooks **in order**:

1. **`notebooks/EDA_and_Preprocessing.ipynb`**  
   Explores the data, fits the preprocessor, writes the processed CSVs and `models/preprocessor.json`.

2. **`notebooks/Modeling_and_Evaluation.ipynb`**  
   Trains XGBoost with CV, calibrates probabilities, tunes thresholds, and saves:
   - `models/xgb.json`
   - `models/calibrator.pkl`
   - `models/threshold.json`

3. **`test/test_model.ipynb`**  
   Scores the unlabeled `test.csv` and produces `data/xgb_submission.csv`.

### Scoring new data in Python

```python
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from notebooks.preprocessing import CreditPreprocessor

# Load artifacts
pre = CreditPreprocessor.load("models/preprocessor.json")
model = XGBClassifier()
model.load_model("models/xgb.json")
calibrator = joblib.load("models/calibrator.pkl")
cfg = json.loads(open("models/threshold.json").read())

# Prepare a single (or batch) row with the original Kaggle columns
raw = pd.read_csv("data/test.csv")          # or your own DataFrame
features = pre.transform(raw)[cfg["feature_columns"]]

# Raw score → calibrated probability
score = model.predict_proba(features)[:, 1]
proba = calibrator.predict(score)

# Optional hard decision
threshold = cfg["threshold_cost"]           # or cfg["threshold_f1"]
is_high_risk = proba >= threshold
```

---

## Streamlit App

An interactive demo is available at `app/app.py`.

```bash
streamlit run app/app.py
```

Features:
- Enter borrower details (age, income, utilization, late-payment history, …)
- Choose **Cost-based** or **F1-based** decision threshold
- See calibrated probability of serious delinquency and a High / Low Risk decision
- Inspect the exact feature vector sent to the model

If you open the project in a GitHub Codespace / VS Code Dev Container, Streamlit starts automatically on port **8501**.

---

## Notes and Limitations

- **Moderate positive-class F1 (~0.43)** reflects the inherent difficulty and class imbalance of the problem. Future improvements: hyper-parameter search, cost-sensitive learning, or external data.
- **Age as a feature:** using age (or proxies for protected attributes) may be restricted by regulation. A fairness review is required before any real-world deployment.
- **Demo only:** the Streamlit app and the model are for educational / demonstration purposes and must not be used as a production lending decision engine.
- **Model format:** the final model is saved in XGBoost’s native JSON format (`xgb.json`) for portability; the calibrator is a small `joblib` pickle.

---

## Acknowledgements

- Dataset: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), Kaggle
- Built with **pandas**, **NumPy**, **scikit-learn**, **XGBoost**, **matplotlib**, **seaborn**, and **Streamlit**
