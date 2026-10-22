"""Load per-project XGBoost artifacts and run inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"

# Map Gerrit project names → short names
PROJECT_MAP: dict[str, str] = {
    "openstack/nova":      "nova",
    "openstack/neutron":   "neutron",
    "openstack/cinder":    "cinder",
    "openstack/glance":    "glance",
    "openstack/designate": "designate",
}

_cache: dict[str, dict] = {}


def _load(short: str) -> dict:
    if short not in _cache:
        path = MODELS_DIR / f"model_{short}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No model artifact at {path}")
        _cache[short] = joblib.load(path)
    return _cache[short]


def supported_projects() -> list[str]:
    manifest = MODELS_DIR / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        return [m["project"] for m in data.get("models", [])]
    return list(PROJECT_MAP.keys())


def predict(feature_row: dict, project: str, threshold_type: str = "gmean") -> dict:
    """Run inference for a single Gerrit change.

    Args:
        feature_row: dict from feature_extractor.build_feature_row()
        project:     full project name, e.g. "openstack/nova"
        threshold_type: "gmean" or "f05"

    Returns a dict with:
        probability, verdict, threshold_used, top_features, train_auc,
        train_mcc_gm, hp_rate, top_reviewers, matched_category
    """
    short = PROJECT_MAP.get(project)
    if short is None:
        raise ValueError(f"Unsupported project: {project!r}.  "
                         f"Supported: {list(PROJECT_MAP.keys())}")

    artifact = _load(short)
    builder  = artifact["builder"]
    model    = artifact["model"]
    thrs     = artifact["thresholds"]

    # SubmissionFeatureBuilder.transform() expects a DataFrame
    df = pd.DataFrame([feature_row])

    X = builder.transform(df)
    prob = float(model.predict_proba(X)[0, 1])

    threshold = float(thrs.get(threshold_type, thrs.get("gmean", 0.5)))
    verdict   = "HP+" if prob >= threshold else "Normal"

    # Feature importances for the risk dashboard
    feature_names  = artifact.get("feature_names", [])
    importances    = model.feature_importances_
    top_n          = 10
    top_idx        = np.argsort(importances)[::-1][:top_n]
    top_features   = [
        {
            "name":       feature_names[i] if i < len(feature_names) else f"feat_{i}",
            "importance": round(float(importances[i]), 4),
        }
        for i in top_idx
    ]

    # Taxonomy category matching
    matched_category = _match_category(feature_row)

    return {
        "probability":     round(prob, 4),
        "verdict":         verdict,
        "threshold_used":  round(threshold, 4),
        "threshold_type":  threshold_type,
        "top_features":    top_features,
        "train_auc":       round(artifact.get("train_auc", 0.0), 3),
        "train_mcc_gm":    round(artifact.get("train_mcc_gm", 0.0), 3),
        "hp_rate":         round(artifact.get("hp_rate", 0.0), 4),
        "top_reviewers":   artifact.get("top_reviewers", []),
        "matched_category": matched_category,
        "n_train":         artifact.get("n_train", 0),
    }


# 26-category taxonomy → human-readable buckets (simplified to 5 for dashboard)
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Release/Maintenance",  ["release_or_milestone_keyword", "is_stable_backport"]),
    ("Bug Fix",              ["is_fix_commit", "is_security_or_cve"]),
    ("Feature/Enhancement",  ["is_feat_commit", "is_blueprint"]),
    ("Infrastructure/CI",    ["is_ci_infra", "is_test_commit"]),
    ("Documentation",        ["is_docs_commit", "is_refactor_commit"]),
]


def _match_category(row: dict) -> str:
    for cat, keys in _CATEGORY_RULES:
        if any(row.get(k, 0) for k in keys):
            return cat
    return "General"
