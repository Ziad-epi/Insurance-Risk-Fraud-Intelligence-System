"""SHAP explainability for tree-based models."""

from pathlib import Path
import matplotlib.pyplot as plt
import shap

from src.utils.helpers import save_figure


def generate_shap_plots(model, X, output_dir="reports/figures"):
    """Generate SHAP summary and feature importance plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap.summary_plot(shap_values, X, show=False)
    fig1 = plt.gcf()
    save_figure(fig1, output_dir / "shap_summary.png")
    plt.close(fig1)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    fig2 = plt.gcf()
    save_figure(fig2, output_dir / "shap_feature_importance.png")
    plt.close(fig2)
