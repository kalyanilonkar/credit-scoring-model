#  Credit Scoring Model for Unsecured Loans

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

A machine learning pipeline for predicting the probability of default on unsecured loan applications.  
Developed as part of the MSc Data Analytics for Business Decision Making (BMAN60422) at the University of Manchester.

---

##  Project Overview
-  **Data:** 5,960 historical loan applications
-  **Goal:** Predict likelihood of default to help banks balance growth & risk
-  **Scenarios:**
  - **Goal 1:** Maximize approvals while catching ≥85% of defaulters
  - **Goal 2:** Be highly risk-averse (≥70% specificity), capturing most defaulters
  - **No goal:** Balanced growth, interpretable decisions

-  Structured under **CRISP-DM framework**
-  Interpretability via **SHAP** & SAS Decision Tree

---

## How to Run
```bash
pip install -r requirements.txt
python EDA.py
python modeling.py
```
- Ensure `imputed_data.xlsx` is in the same folder.

---

## 📂 Files

| File                    | Description                                       |
|--------------------------|---------------------------------------------------|
| `EDA.py`                 | Data cleaning, feature engineering, MI & Lasso    |
| `modeling.py`            | ML pipeline, cross-val, thresholds, SHAP          |
| `FullDecisionTree.svg`   | Decision Tree visualised in SAS                   |
| `imputed_data.xlsx`      | Final cleaned dataset with missing flags & imputes|
| `BMAN60422_Report.pdf`   | Full methodology, results & business insights     |

---

## 🔬 Models & Techniques
- Logistic Regression, Random Forest, HGBT, SVC, KNN, MLP, Decision Tree
- Stratified train-test splits (75:25), 10-fold cross-validation
- Hyperparameter tuning & threshold adjustments
- SHAP values for interpretability
- Business-oriented metrics: Sensitivity, Specificity, PR-AUC

---

## 📈 Business Recommendations
- **Selective growth:** HGBT, balancing approvals with high quality
- **Highly risk-averse:** RF, catching nearly all defaulters, conservative
- **Balanced:** DT, interpretable decisions for compliance (GDPR/FCA)

---

## 👥 Authors
Kalyani Lonkar and Group 8 (2025), University of Manchester  
Course: BMAN60422 - Data Analytics for Business Decision Making

---

## 📌 Keywords
`python` `machine-learning` `credit-risk` `shap` `sas` `sklearn` `business-analytics`