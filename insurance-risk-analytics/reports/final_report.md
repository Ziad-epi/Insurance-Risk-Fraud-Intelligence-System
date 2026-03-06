# Project Overview
This project delivers a full insurance risk analytics pipeline covering claim frequency, claim severity, pure premium estimation, fraud detection, and explainability. It is designed to mirror production workflows used in large insurers for pricing and risk oversight.

# Dataset Description
The project uses policy-level and claim-level datasets. Policy records include exposure, premium, and risk attributes. Claim records include claim amounts, dates, and fraud indicators where available.

# Methodology
The pipeline performs data cleaning, feature engineering, and categorical encoding. Frequency is modeled with a Poisson GLM, severity with a Gamma GLM, and fraud detection with a Random Forest classifier using SMOTE to address imbalance.

# Models Used
- Poisson GLM for claim frequency
- Gamma GLM for claim severity
- Random Forest for fraud detection
- SHAP for model explainability

# Results
Models are evaluated using MAE and RMSE for regression tasks, and ROC AUC with a confusion matrix for fraud classification. Results are logged during execution and can be exported to reporting artifacts.

# Business Insights
The expected loss ratio and pure premium outputs provide a basis for pricing decisions. Fraud probability highlights high-risk claims for investigation. Segmentation allows portfolio managers to target profitable segments and manage exposure.

# Conclusion
The project provides a modular, production-ready framework for insurance risk analytics, enabling rapid experimentation, transparent decisioning, and deployment-ready insights.
