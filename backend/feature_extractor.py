"""Transform a Gerrit change dict → feature dict compatible with SubmissionFeatureBuilder."""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone

from db import lookup_author

# ─── keyword patterns ──────────────────────────────────────────────────────
_RE_BLUEPRINT  = re.compile(r"\bblueprint\b", re.I)
_RE_BACKPORT   = re.compile(r"\bbackport\b", re.I)
_RE_REVERT     = re.compile(r"\brevert\b", re.I)
_RE_CI_INFRA   = re.compile(r"\b(zuul|tox|jenkins|ci|gate|check)\b", re.I)
_RE_SECURITY   = re.compile(r"\b(cve|security|vuln|exploit)\b", re.I)
_RE_TEST       = re.compile(r"\b(test|spec|unit|functional)\b", re.I)
_RE_FIX        = re.compile(r"\b(fix|bug|repair|correct|resolv)\b", re.I)
_RE_REFACTOR   = re.compile(r"\b(refactor|clean|lint|format|tidy)\b", re.I)
_RE_FEAT       = re.compile(r"\b(add|new|support|implement|introduc|feat)\b", re.I)
_RE_DOCS       = re.compile(r"\b(doc|readme|comment|changelog|release.note)\b", re.I)
_RE_RELEASE_KW = re.compile(r"\b(release|milestone|rc\d|alpha|beta)\b", re.I)
_RE_MAINT_KW   = re.compile(r"\b(maintainer|urgent|priority|critical|blocker|sev)\b", re.I)
_RE_HASHTAG    = re.compile(r"#\w+")
_RE_CONV_PREF  = re.compile(r"^(fix|feat|chore|docs|style|refactor|test|perf)\(", re.I)
_RE_DB_REF     = re.compile(r"\b(migration|schema|alembic|db)\b", re.I)

_STABLE_BRANCHES = {"stable/", "unmaintained/"}
_SIZE_THRESHOLDS = [(0, "XS"), (10, "S"), (50, "M"), (250, "L"), (1000, "XL")]


def _size_label(churn: int) -> str:
    label = "XXL"
    for limit, lbl in _SIZE_THRESHOLDS:
        if churn <= limit:
            label = lbl
            break
    return label


def _branch_type(branch: str) -> str:
    if any(branch.startswith(p) for p in _STABLE_BRANCHES):
        return "stable"
    if branch in ("master", "main"):
        return "master"
    return "feature"


def _topic_category(topic: str) -> str:
    if not topic:
        return "none"
    if _RE_BLUEPRINT.search(topic):
        return "blueprint"
    if _RE_BACKPORT.search(topic):
        return "backport"
    return "other"


def _parse_created(created_str: str) -> datetime | None:
    if not created_str:
        return None
    try:
        # "2024-01-15 10:23:00.000000000"
        s = created_str[:19]  # trim nanoseconds
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# Approximate next OpenStack release dates (month-day only matters for proximity)
_RELEASE_DATES = [
    datetime(2024, 3, 20, tzinfo=timezone.utc),   # Caracal
    datetime(2024, 10, 9, tzinfo=timezone.utc),    # Dalmatian
    datetime(2025, 4, 2, tzinfo=timezone.utc),     # Epoxy
    datetime(2025, 10, 8, tzinfo=timezone.utc),    # Flamingo
]


def _days_to_nearest_release(created: datetime | None) -> float:
    if created is None:
        return 180.0
    deltas = [abs((r - created).days) for r in _RELEASE_DATES]
    return float(min(deltas)) if deltas else 180.0


def build_feature_row(change: dict, override_author: dict | None = None) -> dict:
    """Return a one-row dict matching the columns expected by SubmissionFeatureBuilder.

    change: output of gerrit_client.extract_change_info()
    override_author: explicit author history values (e.g. from SQLite lookup);
                     if None we look up in the SQLite author_profiles table.
    """
    subject = change.get("subject", "") or ""
    msg     = change.get("commit_message", "") or subject
    topic   = change.get("topic", "") or ""
    branch  = change.get("branch", "") or ""

    insertions = int(change.get("insertions", 0) or 0)
    deletions  = int(change.get("deletions", 0)  or 0)
    churn      = insertions + deletions

    created = _parse_created(change.get("created", ""))

    # Author history — from SQLite cache or override
    project   = change.get("project", "")
    owner_id  = change.get("owner_id", "")
    if override_author is not None:
        auth = override_author
    else:
        auth = lookup_author(owner_id, project) or {}

    row: dict = {
        # Structural
        "project":   project,
        "branch":    branch,
        "subject":   subject,
        "topic":     topic,

        # Numeric submission features
        "insertions":    insertions,
        "deletions":     deletions,
        "churn_total":   churn,

        "hashtags_count":  len(_RE_HASHTAG.findall(msg)),
        "has_topic":       int(bool(topic)),

        "created_dayofweek": created.weekday() if created else 0,
        "created_hour":      created.hour      if created else 0,
        "created_month":     created.month     if created else 1,

        "days_to_nearest_release":    _days_to_nearest_release(created),
        "has_release_keyword":        int(bool(_RE_RELEASE_KW.search(msg))),
        "is_blueprint":               int(bool(_RE_BLUEPRINT.search(msg))),
        "is_stable_backport":         int(bool(_RE_BACKPORT.search(msg)) or branch.startswith("stable/")),
        "is_revert":                  int(bool(_RE_REVERT.search(msg))),
        "release_or_milestone_keyword": int(bool(_RE_RELEASE_KW.search(msg))),
        "maintainer_attention_keyword": int(bool(_RE_MAINT_KW.search(msg))),
        "is_ci_infra":                int(bool(_RE_CI_INFRA.search(msg))),
        "is_security_or_cve":         int(bool(_RE_SECURITY.search(msg))),
        "is_test_commit":             int(bool(_RE_TEST.search(msg))),
        "is_fix_commit":              int(bool(_RE_FIX.search(msg))),
        "is_refactor_commit":         int(bool(_RE_REFACTOR.search(msg))),
        "is_feat_commit":             int(bool(_RE_FEAT.search(msg))),
        "is_docs_commit":             int(bool(_RE_DOCS.search(msg))),
        "has_conv_prefix":            int(bool(_RE_CONV_PREF.match(subject))),
        "is_db_refactoring":          int(bool(_RE_DB_REF.search(msg))),
        "subject_word_count":         len(subject.split()),
        "subject_len":                len(subject),

        # Categorical
        "branch_type":    _branch_type(branch),
        "size_label":     _size_label(churn),
        "topic_category": _topic_category(topic),

        # Author history (from SQLite or zeros for unknown authors)
        "prior_submission_count":   float(auth.get("prior_submission_count", 0)),
        "prior_merged_count":       float(auth.get("prior_merged_count", 0)),
        "prior_abandoned_count":    float(auth.get("prior_abandoned_count", 0)),
        "prior_hp_count":           float(auth.get("prior_hp_count", 0)),
        "prior_hp_rate":            float(auth.get("prior_hp_rate", 0)),
        "prior_cr_plus2_count":     float(auth.get("prior_cr_plus2_count", 0)),
        "days_since_first_contribution": float(auth.get("days_since_first_contribution", 0)),
        "prior_90d_submissions":    float(auth.get("prior_90d_submissions", 0)),
        "prior_365d_submissions":   float(auth.get("prior_365d_submissions", 0)),
        "prior_merged_ratio":       float(auth.get("prior_merged_ratio", 0)),
        "prior_abandoned_ratio":    float(auth.get("prior_abandoned_ratio", 0)),
        "prior_90d_merged":         float(auth.get("prior_90d_merged", 0)),
        "prior_365d_merged":        float(auth.get("prior_365d_merged", 0)),
        "is_core_developer":        float(auth.get("is_core_developer", 0)),
        "prior_hp_assigned_count":  float(auth.get("prior_hp_assigned_count", 0)),
        "prior_hp_assigned_rate":   float(auth.get("prior_hp_assigned_rate", 0)),
        "prior_cr_plus2_rate":      float(auth.get("prior_cr_plus2_rate", 0)),
        "prior_90d_submission_velocity": float(auth.get("prior_90d_submission_velocity", 0)),
        "prior_submission_recency_ratio": float(auth.get("prior_submission_recency_ratio", 0)),
        "reputation_score":         float(auth.get("reputation_score", 0)),
        "log_prior_submission_count": float(auth.get("log_prior_submission_count",
                                                      math.log1p(auth.get("prior_submission_count", 0)))),
        "log_prior_cr_plus2_count":   float(auth.get("log_prior_cr_plus2_count",
                                                      math.log1p(auth.get("prior_cr_plus2_count", 0)))),
        "log_prior_hp_assigned_count": float(auth.get("log_prior_hp_assigned_count",
                                                       math.log1p(auth.get("prior_hp_assigned_count", 0)))),
        "prior_cr_minus2_count":    float(auth.get("prior_cr_minus2_count", 0)),
        "prior_avg_patchset_count": float(auth.get("prior_avg_patchset_count", 0)),
        "prior_avg_churn":          float(auth.get("prior_avg_churn", 0)),
        "prior_avg_review_latency_h": float(auth.get("prior_avg_review_latency_h", 0)),
        "prior_submission_days_std": float(auth.get("prior_submission_days_std", 0)),
        "global_prior_submissions":  float(auth.get("global_prior_submissions", 0)),
        "global_prior_merged":       float(auth.get("global_prior_merged", 0)),
        "global_prior_hp_count":     float(auth.get("global_prior_hp_count", 0)),
        "global_prior_unique_projects": float(auth.get("global_prior_unique_projects", 0)),

        # Owner hash bucket placeholder (SubmissionFeatureBuilder hashes owner_id internally)
        "owner_id": owner_id,

        # Interaction feature expected by ColumnTransformer (fitted during training)
        "churn_x_is_core": churn * float(auth.get("is_core_developer", 0)),

        # Dummy label column (builder expects it during transform even in inference)
        "hp_positive": 0,
    }
    return row
