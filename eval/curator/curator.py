"""Learned curator — predict the SG retrieval mode for a query.

Loads a trained bundle (TF-IDF vectorizer + LogisticRegression + label list)
and maps a query string to a QueryMode name, used as
`engine.heuristic_query(query, mode_hint=<name>)`. If no model file exists it
returns None, so the `sg-learned` arm transparently falls back to the
rule-based router (and is then identical to `sg`). See docs/CURATOR.md.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = Path(__file__).resolve().parent / "curator_model.pkl"

# process-cache: {model_path: bundle-or-None}
_CACHE: dict = {}


def _load(model_path: Path):
    key = str(model_path)
    if key not in _CACHE:
        p = Path(model_path)
        if not p.is_file():
            _CACHE[key] = None
        else:
            with open(p, "rb") as f:
                _CACHE[key] = pickle.load(f)   # {"vectorizer", "clf", "labels"}
    return _CACHE[key]


def predict_mode(query: str, model_path: Path = DEFAULT_MODEL) -> Optional[str]:
    """Return the predicted QueryMode name, or None if no model is available."""
    bundle = _load(model_path)
    if not bundle or not query:
        return None
    try:
        X = bundle["vectorizer"].transform([query])
        return str(bundle["clf"].predict(X)[0])
    except Exception:
        return None
