# Credit Card Fraud Detection
### Handling Severely Imbalanced Datasets in a Fintech Context

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Business Problem

A European card issuer processes hundreds of thousands of transactions daily. A tiny fraction — **0.17%** — are fraudulent. The challenge is not simply detecting fraud; any model that flags every transaction would catch 100% of cases but be completely unusable in practice.

The real question is:

> **Can we build a model that catches the majority of fraudulent transactions while keeping false positive alerts low enough that an operations team can actually act on them?**

This project works through that problem end-to-end — from understanding the class imbalance, to engineering a reliable model, to evaluating it on metrics that reflect real business cost.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 over two days (September 2013) |
| Features | 30 total: `Time`, `Amount`, and V1–V28 (PCA-transformed, anonymised) |
| Fraud cases | 492 out of 284,807 (0.17%) |
| Imbalance ratio | ~578:1 (legitimate to fraud) |

> **Note on V1–V28:** These features have been PCA-transformed for confidentiality. Their business meaning cannot be interpreted directly, but they carry strong discriminative signal for fraud detection.

---

## Why Standard Accuracy Fails Here

A model that predicts "not fraud" for every single transaction would score **99.83% accuracy** while catching **zero fraud cases**. This project uses metrics that reflect actual business reality:

- **Recall** — what proportion of real fraud cases did we catch?
- **Precision** — of all transactions we flagged, what proportion were genuinely fraudulent?
- **ROC-AUC** — how well does the model rank a fraud case above a legitimate one?
- **Average Precision** — area under the Precision-Recall curve (most informative for imbalanced data)

---

## Analytical Approach

### 1. Data Profiling
Full inventory of shape, null values, class distribution, and raw feature distributions before any transformation.

### 2. Feature Scaling
Applied **RobustScaler** to `Amount` and `Time` — chosen over StandardScaler because transaction amounts have extreme outliers that would distort mean-based scaling.

### 3. Handling Class Imbalance
Two strategies compared:

- **Random Undersampling** — 492 fraud cases matched with 492 randomly selected legitimate cases (50/50 balance). Fast and interpretable; discards most legitimate data.
- **SMOTE** (Synthetic Minority Oversampling Technique) — generates synthetic fraud samples by interpolating between existing cases in feature space. Retains all legitimate transactions.

> Both techniques are applied **only to the training set**. The test set remains in its natural imbalanced state to simulate real production conditions.

### 4. Correlation Analysis
Correlation analysis performed on the balanced subsample (not the imbalanced full dataset, which would obscure the signal). Key findings:

- **Negatively correlated with fraud** (lower value = more likely fraud): V17, V14, V12, V10
- **Positively correlated with fraud** (higher value = more likely fraud): V11, V4, V2, V19

### 5. Outlier Removal
IQR-based extreme outlier removal applied to the three most discriminative negative features (V14, V12, V10) within the fraud class only. This tightens the model's decision boundary without discarding legitimate transaction data.

### 6. Dimensionality Reduction
t-SNE, PCA, and Truncated SVD used to confirm class separability in 2D — a visual sanity check that the features carry real signal before committing to modelling.

### 7. Model Comparison
Four classifiers evaluated with cross-validation on the balanced training set:
- Logistic Regression (tuned with GridSearchCV)
- K-Nearest Neighbours (tuned with GridSearchCV)
- Support Vector Machine
- Decision Tree

### 8. Final Evaluation
Best model evaluated on the **original imbalanced test set** using confusion matrices, ROC curves, and Precision-Recall curves.

---

## Results

### Model Performance on Original Imbalanced Test Set

| Metric | Undersampling | SMOTE |
|---|---|---|
| Recall | 0.88 | 0.93 |
| Precision | 0.86 | 0.72 |
| F1 Score | 0.87 | 0.81 |
| ROC-AUC | 0.97 | 0.98 |
| Avg Precision | 0.74 | 0.79 |

### Confusion Matrix — Best Model (SMOTE + Logistic Regression)

|  | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 56,848 | 16 |
| **Actual Fraud** | 7 | 91 |

### What This Means in Practice

The SMOTE-trained Logistic Regression correctly flagged **91 out of 98 fraudulent transactions** in the test set — a recall rate of 93%. It generated false alerts on just **16 legitimate transactions** out of 56,864 — meaning the operations team would investigate 107 flagged transactions total, 85% of which are genuine fraud. That is an operationally viable signal-to-noise ratio for a fraud review team.

---

## Key Visualisations

The notebook produces the following visuals:

- Class distribution (original imbalanced dataset)
- Transaction Amount and Time distributions
- Correlation heatmaps — imbalanced vs balanced subsample
- Boxplots of the most discriminative features by class
- t-SNE / PCA / SVD cluster plots confirming class separability
- ROC curves comparing all four classifiers
- Confusion matrices — undersampling vs SMOTE
- Precision-Recall curves — undersampling vs SMOTE

---

## Project Structure

```
credit-fraud-detection/
│
├── credit_fraud_detection.ipynb   # Main analysis notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

> The dataset (`creditcard.csv`) is not included due to size. Download it from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the root directory before running the notebook.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-fraud-detection.git
cd credit-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place creditcard.csv in the root folder

# 4. Launch Jupyter
jupyter notebook credit_fraud_detection.ipynb
```

Alternatively, run directly on [Kaggle](https://www.kaggle.com/) where the dataset is pre-loaded.

---

## Requirements

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.3
imbalanced-learn>=0.11
jupyter
```

---

## Analytical Decisions and Tradeoffs

| Decision | Rationale |
|---|---|
| RobustScaler over StandardScaler | Transaction amounts have extreme outliers; RobustScaler uses IQR and is not distorted by them |
| Evaluate on original imbalanced test set | Reflects true production conditions — sampling only happens in training |
| Recall as primary metric | Missing fraud is more costly than a false alert in most card fraud contexts |
| Compare undersampling and SMOTE | Neither is universally superior; the right choice depends on dataset size and the cost model |
| IQR outlier removal on fraud class only | Avoids distorting the model boundary without touching legitimate transaction data |

---

## Limitations

- **Feature opacity:** V1–V28 are PCA-transformed for confidentiality. The model cannot be explained in business terms — only in statistical ones.
- **Temporal drift:** The dataset covers two days in 2013. Fraud patterns evolve; this model would require periodic retraining in production.
- **Threshold sensitivity:** The default decision threshold (0.5) can be tuned to shift the precision/recall trade-off based on the bank's cost model.
- **Geography:** Data reflects European cardholders. Performance may differ on other populations.

---

## Acknowledgements

Dataset sourced from the [ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud) via Kaggle. Analysis structure informed by community kernels on the same dataset.

---

## About

Built by **Kipngetich** — freelance data analyst specialising in analytics and insight for business decision-making.

· [Portfolio](https://github.com/kipngetich-kk) 
