#!/usr/bin/env python3
"""Train and serialize per-project XGBoost models for HP+ prediction.

Reads data/processed/dataset_with_subjects.csv from the parent research repo,
fits one SubmissionFeatureBuilder + XGBoost pipeline per project, and writes
joblib artifacts to hp-priority-checker/backend/models/.

Usage:
    python scripts/train_models.py
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score, matthews_corrcoef
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOOL_ROOT   = Path(__file__).parent.parent
REPO_ROOT   = TOOL_ROOT.parent / "review-priority-predictor"
DATA_CSV    = REPO_ROOT / "data" / "processed" / "dataset_with_subjects.csv"
MODELS_DIR  = TOOL_ROOT / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SubmissionFeatureBuilder is bundled in backend/rp_predict/
sys.path.insert(0, str(TOOL_ROOT / "backend"))
from rp_predict.features.submission import SubmissionFeatureBuilder  # noqa: E402

# Projects we build models for
PROJECTS = [
    "openstack/nova",
    "openstack/neutron",
    "openstack/cinder",
    "openstack/glance",
    "openstack/designate",
]

# Short names for filenames
SHORT = {
    "openstack/nova":       "nova",
    "openstack/neutron":    "neutron",
    "openstack/cinder":     "cinder",
    "openstack/glance":     "glance",
    "openstack/designate":  "designate",
}

# Top HP+ reviewers per project (from hp_voter_emails.csv analysis)
TOP_REVIEWERS = {
    "openstack/nova":       ["rodolfo.alonso.hernandez@gmail.com", "sean.k.mooney@gmail.com"],
    "openstack/neutron":    ["rodolfo.alonso.hernandez@gmail.com", "sean.k.mooney@gmail.com"],
    "openstack/cinder":     ["brian.rosmaita@gmail.com", "walter.boring@gmail.com"],
    "openstack/glance":     ["brian.rosmaita@gmail.com"],
    "openstack/designate":  [],
}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns that SubmissionFeatureBuilder expects but dataset doesn't have."""
    df = df.copy()
    # log-scale of prior HP+ assigned count (prevents very large values dominating)
    if "prior_hp_assigned_count" in df.columns and "log_prior_hp_assigned_count" not in df.columns:
        df["log_prior_hp_assigned_count"] = np.log1p(df["prior_hp_assigned_count"].fillna(0))
    return df


def find_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Find G-mean optimal threshold and F0.5 optimal threshold."""
    fpr, tpr, thrs = roc_curve(y_true, y_prob)
    gmean = np.sqrt(tpr * (1 - fpr))
    thr_gm = float(thrs[np.argmax(gmean)])

    best_f05, thr_fp = -1.0, 0.5
    for t in np.unique(thrs):
        yp = (y_prob >= t).astype(int)
        if yp.sum() == 0:
            continue
        f05 = float(fbeta_score(y_true, yp, beta=0.5, zero_division=0))
        if f05 > best_f05:
            best_f05, thr_fp = f05, float(t)

    return {"gmean": thr_gm, "f05": thr_fp}


def make_model(scale_pos_weight: float) -> XGBClassifier:
    """XGBoost settings shared by training and scripts/evaluate_holdout.py."""
    # Same hyperparams as in the dynamic eval
    return XGBClassifier(
        n_estimators        = 300,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        scale_pos_weight    = scale_pos_weight,
        use_label_encoder   = False,
        eval_metric         = "logloss",
        random_state        = 42,
        n_jobs              = -1,
        verbosity           = 0,
    )


def train_project(project: str, df_project: pd.DataFrame) -> dict:
    short = SHORT[project]
    print(f"\n{'='*60}")
    print(f"  {short}  ({len(df_project)} rows, {int(df_project['hp_positive'].sum())} HP+)")
    print(f"{'='*60}")

    df_project = add_derived_columns(df_project)

    # Compute class imbalance ratio for XGBoost scale_pos_weight
    n_neg = int((df_project["hp_positive"] == 0).sum())
    n_pos = int((df_project["hp_positive"] == 1).sum())
    spw   = round(n_neg / max(n_pos, 1), 2)
    print(f"  neg={n_neg}  pos={n_pos}  scale_pos_weight={spw}")

    builder = SubmissionFeatureBuilder()
    X, y = builder.fit_transform(df_project)
    print(f"  Feature matrix: {X.shape}")

    model = make_model(spw)
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    auc    = float(roc_auc_score(y, y_prob))
    thrs   = find_thresholds(y, y_prob)

    yp_gm = (y_prob >= thrs["gmean"]).astype(int)
    mcc_gm = float(matthews_corrcoef(y, yp_gm)) if yp_gm.sum() > 0 else 0.0

    print(f"  Train AUC={auc:.3f}  MCC(G-mean thr)={mcc_gm:.3f}")
    print(f"  Threshold G-mean={thrs['gmean']:.4f}  F0.5={thrs['f05']:.4f}")

    # Feature names for dashboard
    feature_names = builder.get_feature_names()

    # Top feature importances (for the risk dashboard)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    top_features = [
        {"name": feature_names[i] if i < len(feature_names) else f"feat_{i}",
         "importance": float(importances[i])}
        for i in top_idx
    ]

    # Author HP+ base rates from training data (for LLM prompt context)
    hp_rate = float(df_project["hp_positive"].mean())

    # Save artifact
    artifact = {
        "project":        project,
        "short":          short,
        "builder":        builder,
        "model":          model,
        "thresholds":     thrs,
        "train_auc":      auc,
        "train_mcc_gm":   mcc_gm,
        "hp_rate":        hp_rate,
        "n_train":        len(df_project),
        "n_pos":          n_pos,
        "scale_pos_weight": spw,
        "feature_names":  feature_names,
        "top_features":   top_features,
        "top_reviewers":  TOP_REVIEWERS.get(project, []),
    }

    out_path = MODELS_DIR / f"model_{short}.joblib"
    joblib.dump(artifact, out_path)
    print(f"  Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
    return {"project": project, "auc": auc, "mcc_gm": mcc_gm,
            "thresholds": thrs, "hp_rate": hp_rate}


def main():
    print(f"Loading dataset from {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    summary = []
    for project in PROJECTS:
        df_proj = df[df["project"] == project].copy()
        if len(df_proj) == 0:
            print(f"  WARNING: no rows for {project}, skipping")
            continue
        if df_proj["hp_positive"].sum() < 10:
            print(f"  WARNING: only {df_proj['hp_positive'].sum()} HP+ for {project}, skipping")
            continue
        result = train_project(project, df_proj)
        summary.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Project':30s} {'AUC':>7} {'MCC':>7} {'HP%':>6}")
    for r in summary:
        print(f"  {r['project']:28s} {r['auc']:7.3f} {r['mcc_gm']:7.3f} {r['hp_rate']*100:5.1f}%")

    # Write a JSON manifest listing available models
    manifest = {
        "models": [
            {"project": r["project"], "short": SHORT[r["project"]],
             "file": f"model_{SHORT[r['project']]}.joblib",
             "train_auc": r["auc"], "hp_rate": r["hp_rate"],
             "thresholds": r["thresholds"]}
            for r in summary
        ]
    }
    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
