# Credit Risk Prediction — *Give Me Some Credit*

A machine learning project that predicts the probability that a borrower will experience **serious financial distress (90+ days past due) within the next two years**. It covers the full workflow: exploratory data analysis, reusable preprocessing, model comparison, threshold tuning, and inference on unseen data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
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
- [Notes and Limitations](#notes-and-limitations)
- [Acknowledgements](#acknowledgements)

---

## Overview

Banks rely on credit scoring to decide whether to approve a loan. This project builds a model that, given a borrower's financial profile, outputs a **probability of default** (`SeriousDlqin2yrs = 1`).

The main challenge is **class imbalance**: only about **6.6%** of borrowers in the data are positive cases, so plain accuracy is misleading. Model selection is therefore based on **ROC-AUC**, and the decision threshold is tuned for **F1 on the positive class** instead of using the default 0.5.

## Dataset

- **Source:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) on Kaggle.
- **Target:** `SeriousDlqin2yrs` — 1 if the person experienced 90 days past due delinquency or worse, otherwise 0.
- **Size used here:** `train.csv` has 104,805 labeled rows and `test.csv` has 45,195 unlabeled rows (together 150,000, matching the size of the original Kaggle training file).

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Balance on credit cards and personal lines divided by the sum of credit limits |
| `age` | Age of the borrower in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days past due (not worse) in the last 2 years |
| `DebtRatio` | Monthly debt payments divided by monthly gross income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and lines of credit |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days past due (not worse) in the last 2 years |
| `NumberOfDependents` | Number of dependents, excluding the borrower |

> The raw data is not redistributed with a license of its own here. Please download it from Kaggle and review the competition's terms before reuse.

## Project Structure

```
My_First_ML_Model/
├── data/
│   ├── train.csv                 # Raw labeled data (Kaggle)
│   ├── test.csv                  # Raw unlabeled data (Kaggle)
│   ├── X_train_final.csv         # Processed training features
│   ├── X_test_final.csv          # Processed hold-out features
│   ├── y_train_raw.csv           # Training labels
│   ├── y_test_raw.csv            # Hold-out labels
│   └── xgb_submission.csv        # Predicted probabilities for test.csv
├── models/
│   ├── best_credit_model.pkl     # Selected model (Random Forest)
│   ├── xgb_credit_model.pkl      # XGBoost model
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── preprocessor.json         # Training medians + final feature order
│   └── threshold.json            # Selected model name + decision threshold
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Modeling_and_Evaluation.ipynb
│   └── preprocessing.py          # Shared CreditPreprocessor class
├── test/
│   └── test_model.ipynb          # Inference on the unlabeled test set
└── requirements.txt
```

## Methodology

### 1. Data splitting
`train.csv` is split into 80% train and 20% hold-out with `train_test_split(..., random_state=42, stratify=y)`, so both parts keep the same positive rate (6.63%).

### 2. Preprocessing (`notebooks/preprocessing.py`)
All cleaning logic lives in one `CreditPreprocessor` class that is **fitted on the training split only** and reused for training, evaluation and inference. This avoids train/serve skew and data leakage.

- **Missing values:** `MonthlyIncome` (~20% missing) and `NumberOfDependents` get binary `...Missing` indicator columns, then are imputed with the **training median**.
- **Invalid sentinel codes:** the values `96` and `98` in the three past-due columns are placeholders, not real counts. They are flagged (`...InvalidFlag`) and replaced by NaN before imputation.
- **Invalid ages:** ages below 18 are treated as missing.
- **Persistence:** the learned medians and final column order are saved to `models/preprocessor.json`, so inference does not depend on pickling a custom class.

### 3. Feature engineering

| Feature | Formula |
|---|---|
| `TotalMonthlyDebt` | `MonthlyIncome × DebtRatio` |
| `IncomePerDependentAdjusted` | `MonthlyIncome / (NumberOfDependents + 1)` |
| `TotalDelinquencies` | Sum of the three past-due counters |
| `RealEstateToTotalCredit` | `NumberRealEstateLoansOrLines / (NumberOfOpenCreditLinesAndLoans + 1)` |

`TotalDelinquencies` turned out to be the feature most correlated with the target (≈ 0.39 on the training split).

### 4. Modeling
Three models are compared on standardized features:

| Model | Imbalance handling |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest (100 trees) | none |
| XGBoost (100 trees, lr = 0.1) | `scale_pos_weight` = negatives / positives |

### 5. Model and threshold selection
A validation set (20% of the training split) is used to choose the model and, for each model, the probability threshold that **maximizes F1 on the positive class** (via the precision–recall curve). The hold-out set is only used for final reporting.

## Results

Evaluated on the 20,961-row hold-out set (1,390 positives), using the validation-tuned threshold of each model:

| Model | ROC-AUC | Threshold | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|---|
| Logistic Regression | 0.8190 | 0.717 | 0.390 | 0.433 | 0.410 |
| Random Forest | 0.8421 | 0.310 | 0.421 | 0.391 | 0.405 |
| **XGBoost** | **0.8615** | 0.772 | 0.398 | 0.468 | 0.430 |

Validation-based selection picked **Random Forest** (threshold 0.31), which is what is stored in `models/best_credit_model.pkl` and `models/threshold.json`. XGBoost achieved the best ROC-AUC and F1 on the hold-out set and is the model used to generate `data/xgb_submission.csv`.

Overall accuracy of the selected Random Forest is 0.92, but because of the class imbalance, the positive-class precision/recall above are the more meaningful numbers.

## Getting Started

### Prerequisites
- Python 3.11 (the project was developed with 3.11)
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd My_First_ML_Model

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install jupyter              # if not already installed
```

### Get the data
Download `cs-training.csv` / `cs-test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) (or use the Kaggle CLI) and place the files in `data/` as `train.csv` and `test.csv` with the column layout described above.

## Usage

Run the notebooks **from inside their own folders** (the first two import `preprocessing` from the same directory), in this order:

1. `notebooks/EDA_and_Preprocessing.ipynb` — explores the data, fits the preprocessor, and writes the processed CSVs and `preprocessor.json`.
2. `notebooks/Modeling_and_Evaluation.ipynb` — trains and compares the models, tunes thresholds, and saves the models, scaler and threshold.
3. `test/test_model.ipynb` — loads the saved artifacts and scores the unlabeled `test.csv`, producing `data/xgb_submission.csv`.

### Scoring new data in Python

```python
import sys, joblib, pandas as pd
sys.path.insert(0, ".")
from notebooks.preprocessing import CreditPreprocessor

pre    = CreditPreprocessor.load("models/preprocessor.json")
scaler = joblib.load("models/scaler.pkl")
model  = joblib.load("models/xgb_credit_model.pkl")

raw = pd.read_csv("data/test.csv")              # same columns as the Kaggle data
X   = scaler.transform(pre.transform(raw))
raw["Probability"] = model.predict_proba(X)[:, 1]
```

To get a hard class label with the saved Random Forest instead, load `best_credit_model.pkl` and `threshold.json` and apply `proba >= threshold`.

## Notes and Limitations

- **Model file size:** `best_credit_model.pkl` is ~87 MB, close to GitHub's 100 MB per-file limit. Consider [Git LFS](https://git-lfs.com/), a smaller forest (e.g. fewer trees or limited depth), or hosting the file elsewhere.
- **Virtual environment:** do not commit `.venv/` — add it to `.gitignore` together with `__pycache__/`.
- **Scaling:** `StandardScaler` is applied to all models for a uniform pipeline, although tree-based models do not require it.
- **Selection vs. hold-out:** Random Forest was chosen on the validation split, while XGBoost scored higher on the hold-out set. Both are kept; for a production decision, a cross-validated comparison would be more robust.
- **Moderate F1:** an F1 of roughly 0.4 on the positive class reflects the difficulty and imbalance of the problem. Future work: hyperparameter tuning, cross-validation, probability calibration, and cost-sensitive threshold selection based on the real cost of a missed default versus a rejected good borrower.
- **Fairness:** the model uses `age` as a feature. In real lending, using age (or proxies for protected attributes) may be restricted by regulation, so a fairness review is required before any real-world use.

## Acknowledgements

- Dataset: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), Kaggle.
- Built with pandas, NumPy, scikit-learn, XGBoost, matplotlib and seaborn.
