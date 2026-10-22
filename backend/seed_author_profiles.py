#!/usr/bin/env python3
"""Pre-populate author_profiles SQLite table from the research dataset.

Run once before starting the Flask backend so that known authors get
historical feature values instead of zeros.

Usage:
    python backend/seed_author_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOOL_ROOT = Path(__file__).parent.parent
REPO_ROOT = TOOL_ROOT.parent / "review-priority-predictor"
DATA_CSV  = REPO_ROOT / "data" / "processed" / "dataset_with_subjects.csv"

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db, upsert_author, AUTHOR_COLS  # noqa: E402

PROJECTS = [
    "openstack/nova", "openstack/neutron", "openstack/cinder",
    "openstack/glance", "openstack/designate",
]


def main():
    print(f"Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    import numpy as np
    if "prior_hp_assigned_count" in df.columns and "log_prior_hp_assigned_count" not in df.columns:
        df["log_prior_hp_assigned_count"] = np.log1p(df["prior_hp_assigned_count"].fillna(0))
    if "prior_submission_count" in df.columns and "log_prior_submission_count" not in df.columns:
        df["log_prior_submission_count"] = np.log1p(df["prior_submission_count"].fillna(0))
    if "prior_cr_plus2_count" in df.columns and "log_prior_cr_plus2_count" not in df.columns:
        df["log_prior_cr_plus2_count"] = np.log1p(df["prior_cr_plus2_count"].fillna(0))

    init_db()
    total = 0
    for project in PROJECTS:
        proj_df = df[df["project"] == project].copy()
        if proj_df.empty:
            continue

        # Keep most recent row per owner (latest author history)
        if "created_ts" in proj_df.columns:
            proj_df = proj_df.sort_values("created_ts")
        proj_df = proj_df.groupby("owner_id").last().reset_index()

        available = [c for c in AUTHOR_COLS if c in proj_df.columns]
        n = 0
        for _, row in proj_df.iterrows():
            owner_id = str(row.get("owner_id", ""))
            if not owner_id:
                continue
            features = {c: float(row[c]) for c in available if pd.notna(row.get(c))}
            upsert_author(owner_id, project, features)
            n += 1
        print(f"  {project}: seeded {n} author profiles")
        total += n

    print(f"\nDone — {total} author profiles seeded into hp_checker.db")


if __name__ == "__main__":
    main()
