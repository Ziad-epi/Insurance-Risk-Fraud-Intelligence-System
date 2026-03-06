"""Utility helpers for model and artifact persistence."""

from pathlib import Path
import joblib


def save_model(model, path):
    """Save a trained model to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a model from disk using joblib."""
    return joblib.load(path)


def save_figure(fig, path, dpi=150):
    """Save matplotlib or Plotly figures to disk.

    If a Plotly figure is provided, HTML export is used by default to avoid
    extra dependencies for static image export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(fig, "savefig"):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return

    if hasattr(fig, "write_html"):
        if path.suffix.lower() in {".html", ".htm"}:
            fig.write_html(str(path))
        else:
            fig.write_html(str(path.with_suffix(".html")))
        return

    raise ValueError("Unsupported figure type for saving")
