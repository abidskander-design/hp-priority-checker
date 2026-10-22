"""SQLite layer: prediction log + author profiles cache."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "hp_checker.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            change_id       TEXT NOT NULL,
            project         TEXT NOT NULL,
            subject         TEXT,
            probability     REAL,
            verdict         TEXT,
            threshold_used  REAL,
            threshold_type  TEXT,
            mcc_train       REAL,
            auc_train       REAL,
            matched_category TEXT,
            top_signals     TEXT,
            llm_explanation TEXT,
            latency_ms      REAL
        );

        CREATE TABLE IF NOT EXISTS author_profiles (
            owner_id                    TEXT NOT NULL,
            project                     TEXT NOT NULL,
            prior_submission_count      REAL DEFAULT 0,
            prior_merged_count          REAL DEFAULT 0,
            prior_abandoned_count       REAL DEFAULT 0,
            prior_hp_count              REAL DEFAULT 0,
            prior_hp_rate               REAL DEFAULT 0,
            prior_cr_plus2_count        REAL DEFAULT 0,
            days_since_first_contribution REAL DEFAULT 0,
            prior_90d_submissions       REAL DEFAULT 0,
            prior_365d_submissions      REAL DEFAULT 0,
            prior_merged_ratio          REAL DEFAULT 0,
            prior_abandoned_ratio       REAL DEFAULT 0,
            prior_90d_merged            REAL DEFAULT 0,
            prior_365d_merged           REAL DEFAULT 0,
            is_core_developer           REAL DEFAULT 0,
            prior_hp_assigned_count     REAL DEFAULT 0,
            prior_hp_assigned_rate      REAL DEFAULT 0,
            prior_cr_plus2_rate         REAL DEFAULT 0,
            prior_90d_submission_velocity REAL DEFAULT 0,
            prior_submission_recency_ratio REAL DEFAULT 0,
            reputation_score            REAL DEFAULT 0,
            log_prior_submission_count  REAL DEFAULT 0,
            log_prior_cr_plus2_count    REAL DEFAULT 0,
            log_prior_hp_assigned_count REAL DEFAULT 0,
            prior_cr_minus2_count       REAL DEFAULT 0,
            prior_avg_patchset_count    REAL DEFAULT 0,
            prior_avg_churn             REAL DEFAULT 0,
            prior_avg_review_latency_h  REAL DEFAULT 0,
            prior_submission_days_std   REAL DEFAULT 0,
            global_prior_submissions    REAL DEFAULT 0,
            global_prior_merged         REAL DEFAULT 0,
            global_prior_hp_count       REAL DEFAULT 0,
            global_prior_unique_projects REAL DEFAULT 0,
            PRIMARY KEY (owner_id, project)
        );
        """)


def log_prediction(
    change_id: str,
    project: str,
    subject: str,
    probability: float,
    verdict: str,
    threshold_used: float,
    threshold_type: str,
    mcc_train: float,
    auc_train: float,
    matched_category: str,
    top_signals: str,
    llm_explanation: str,
    latency_ms: float,
) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO predictions
               (ts, change_id, project, subject, probability, verdict,
                threshold_used, threshold_type, mcc_train, auc_train,
                matched_category, top_signals, llm_explanation, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), change_id, project, subject, probability, verdict,
             threshold_used, threshold_type, mcc_train, auc_train,
             matched_category, top_signals, llm_explanation, latency_ms),
        )
        return cur.lastrowid


def get_history(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM predictions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_prediction(change_id: str, project: str, max_age_seconds: int = 300) -> dict | None:
    """Return the most recent prediction for this change if it's younger than max_age_seconds."""
    cutoff = time.time() - max_age_seconds
    with _conn() as con:
        row = con.execute(
            """SELECT * FROM predictions
               WHERE change_id=? AND project=? AND ts>=?
               ORDER BY ts DESC LIMIT 1""",
            (change_id, project, cutoff),
        ).fetchone()
    return dict(row) if row else None


def get_latency_stats() -> dict:
    with _conn() as con:
        row = con.execute(
            """SELECT COUNT(*) as n,
                      AVG(latency_ms) as mean,
                      MIN(latency_ms) as min,
                      MAX(latency_ms) as max
               FROM predictions WHERE latency_ms IS NOT NULL"""
        ).fetchone()
    return dict(row) if row else {}


AUTHOR_COLS = [
    "prior_submission_count", "prior_merged_count", "prior_abandoned_count",
    "prior_hp_count", "prior_hp_rate", "prior_cr_plus2_count",
    "days_since_first_contribution", "prior_90d_submissions",
    "prior_365d_submissions", "prior_merged_ratio", "prior_abandoned_ratio",
    "prior_90d_merged", "prior_365d_merged", "is_core_developer",
    "prior_hp_assigned_count", "prior_hp_assigned_rate",
    "prior_cr_plus2_rate", "prior_90d_submission_velocity",
    "prior_submission_recency_ratio", "reputation_score",
    "log_prior_submission_count", "log_prior_cr_plus2_count",
    "log_prior_hp_assigned_count", "prior_cr_minus2_count",
    "prior_avg_patchset_count", "prior_avg_churn",
    "prior_avg_review_latency_h", "prior_submission_days_std",
    "global_prior_submissions", "global_prior_merged",
    "global_prior_hp_count", "global_prior_unique_projects",
]


def upsert_author(owner_id: str, project: str, features: dict) -> None:
    cols   = ["owner_id", "project"] + [k for k in AUTHOR_COLS if k in features]
    vals   = [owner_id, project] + [features[k] for k in AUTHOR_COLS if k in features]
    placeholders = ", ".join(["?"] * len(vals))
    col_str      = ", ".join(cols)
    update_str   = ", ".join(f"{c}=excluded.{c}" for c in cols[2:])
    with _conn() as con:
        con.execute(
            f"INSERT INTO author_profiles ({col_str}) VALUES ({placeholders}) "
            f"ON CONFLICT(owner_id, project) DO UPDATE SET {update_str}",
            vals,
        )


def lookup_author(owner_id: str, project: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM author_profiles WHERE owner_id=? AND project=?",
            (owner_id, project),
        ).fetchone()
    return dict(row) if row else None
