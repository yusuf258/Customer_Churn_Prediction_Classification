# Customer Churn Prediction | Binary Classification

Classification pipeline to predict which telecom customers are likely to cancel their subscription, enabling proactive retention strategies.

## Problem Statement
Predict customer **churn** (Yes/No) based on account details, service usage, and contract information. Early identification of at-risk customers enables targeted retention campaigns.

## Dataset
| Attribute | Detail |
|---|---|
| File | `Telco-Customer-Churn.csv` |
| Records | 7,043 customers |
| Features | 21 (tenure, contract type, monthly charges, services, etc.) |
| Target | `Churn` (Yes / No) |
| Class Balance | ~73.5% No / ~26.5% Yes |

## Methodology
1. **EDA & Visualization** — Churn rate by contract type, tenure distribution, service usage analysis
2. **Feature Engineering** — `TotalCharges` conversion, tenure grouping
3. **Preprocessing** — `StandardScaler` + `OneHotEncoder` / `OrdinalEncoder` via `ColumnTransformer`
4. **ML Models** — Logistic Regression, Random Forest, Gradient Boosting, XGBoost
5. **DL Model** — Dense Neural Network with Dropout and `EarlyStopping`
6. **Evaluation** — Accuracy, F1-Score, ROC-AUC, Confusion Matrix

## Results
| Model | Accuracy |
|---|---|
| Multiple ML models | ~78–79% range |
| **Best Model** | Selected automatically by highest accuracy |

> Monthly contract customers and those with fiber optic internet show highest churn rates.

## Technologies
`Python` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `joblib`

## File Structure
```
12_Customer_Churn_Prediction_Classification/
├── project_notebook.ipynb          # Main notebook
├── Telco-Customer-Churn.csv        # Dataset
└── models/                         # Saved model files
```

## How to Run
```bash
cd 12_Customer_Churn_Prediction_Classification
jupyter notebook project_notebook.ipynb
```
