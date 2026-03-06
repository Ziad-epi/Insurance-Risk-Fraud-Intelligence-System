# Insurance Risk & Fraud Intelligence System

End‑to‑end insurance risk analytics covering pricing (frequency & severity), fraud detection, explainability (SHAP), and an interactive Streamlit dashboard.

## What’s Inside
- **Modeling pipeline**: GLM frequency/severity, pure premium, fraud model.
- **Explainability**: SHAP plots for fraud model.
- **Dashboard**: Streamlit app to explore pricing, risk segmentation, and fraud insights.
- **Data workflow**: raw → processed → reports/figures.

## Repository Structure
- `insurance-risk-analytics/` Core ML pipeline, config, notebooks, and dashboard.
- `Dataset/` Source CSV files used to seed the raw data folder.
- `streamlit_app.py` Root Streamlit dashboard entrypoint.

Key folders in `insurance-risk-analytics/`:
- `data/raw/` Raw input CSVs.
- `data/processed/` Pipeline outputs (`modeling_dataset.csv`).
- `models/` Trained model artifacts.
- `reports/figures/` SHAP and other plots.
- `src/` Reusable data, feature, and model code.
- `config/` Project configuration.

## Data Placement
Expected raw CSVs live in:
```
insurance-risk-analytics/data/raw/
    policies.csv
    customers.csv
    claims.csv
    payments.csv  (optional, if available)
```

If your source files are in `Dataset/`, copy them once:
```powershell
Copy-Item Dataset\policies.csv, Dataset\customers.csv, Dataset\claims.csv `
  -Destination insurance-risk-analytics\data\raw\ -Force
```

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r insurance-risk-analytics\requirements.txt
```

## Run the Pipeline
```powershell
cd insurance-risk-analytics
python main.py
```

Outputs:
- `insurance-risk-analytics/data/processed/modeling_dataset.csv`
- `insurance-risk-analytics/models/*.pkl`
- `insurance-risk-analytics/reports/figures/*.png`

## Run the Dashboard
Root app (recommended; auto‑detects processed data):
```powershell
python -m streamlit run streamlit_app.py
```

Alternative dashboard inside the subproject:
```powershell
python -m streamlit run insurance-risk-analytics\dashboard\app.py
```

## Notes
- The root Streamlit app prefers `insurance-risk-analytics/data/processed/modeling_dataset.csv` but can fall back to raw policy data if needed.
- If you don’t have `payments.csv`, the pipeline still runs.

## Troubleshooting
- **`ModuleNotFoundError: imblearn`** → install deps: `pip install -r insurance-risk-analytics\requirements.txt`
- **Streamlit can’t find data** → ensure `modeling_dataset.csv` exists or raw CSVs are in `insurance-risk-analytics/data/raw/`
