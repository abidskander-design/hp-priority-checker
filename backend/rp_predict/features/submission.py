"""Submission-time feature builder — strictly no post-submission leakage.

Every feature in this module is observable at the moment a change is uploaded
(or uses only information from *prior* changes by the same author).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from rp_predict.utils.logging import get_logger
from rp_predict.utils.release_dates import days_to_nearest_release

log = get_logger(__name__)

# Release-relevant keywords found in commit subjects
_RELEASE_KEYWORDS = re.compile(
    r"\b(rc\d*|release|blocker|critical|urgent|regression|backport|"
    r"milestone|freeze|stable[\-_]only)\b",
    re.IGNORECASE,
)
_BLUEPRINT_PREFIX = re.compile(r"^(bp[-_/]|\[?blueprint)", re.IGNORECASE)

# ── Taxonomy-derived subject patterns ────────────────────────────────────────
_STABLE_BACKPORT   = re.compile(r"\[stable|\bstable[- ]only\b|\bbackport\b|cherry[- ]pick", re.I)
_REVERT            = re.compile(r"^revert[\s\"]", re.I)
_DB_REFACTOR       = re.compile(r"^db[:\s]", re.I)
_OVN_RELATED       = re.compile(r"\bovn\b|\[ovn\b|\[ovn/", re.I)
_RBAC_POLICY       = re.compile(r"\[s-rbac\]|\bs[- ]rbac\b|\[secure rbac\]|secure rbac|new polic", re.I)
_CI_INFRA          = re.compile(r"^(mypy|tox|tests?|zuul|functional|dnm|wip)[:\s\[]|^\[ci\]|^\[ft\]", re.I)
_BRACKET_PREFIX    = re.compile(r"^\s*\[")
_VENDOR_NAMES      = re.compile(
    r"\b(hitachi|netapp|dell|hpe|3par|ibm|pure storage|ceph|storpool|"
    r"fujitsu|datera|lightos|fungible|powermax|powerflex|powerstore|"
    r"nvme|infinidat|solidfire|qnap|synology|zadara)\b",
    re.I,
)
_DRIVER_WORK       = re.compile(
    r"(initial commit for.*driver|new.*driver|add.*driver|"
    r"\bvast data\b|\bnvme.tcp.*driver\b)",
    re.I,
)
_SECURITY_CVE      = re.compile(
    r"\bcve-\d{4}-\d+\b|\bsecurity.fix\b|\bsecurity.issue\b|\bvulnerabilit|\b"
    r"privilege.escalation\b|\bsandbox.escape\b|\btoken.leak|\brce\b|\binjection\b",
    re.I,
)
_MAINTAINER_ATTN   = re.compile(
    r"\burgent\b|\battention\b|\bcritical.bug\b|\bshow.stopper\b|\bregression\b"
    r"|\bdata.loss\b|\bdata.corruption\b|\bmemory.leak\b|\bdeadlock\b",
    re.I,
)
_MILESTONE_KW      = re.compile(
    r"\bmilestone\b|\bfeature.freeze\b|\bdeadline\b|\brelease.candidate\b"
    r"|\bstring.freeze\b|\bcode.freeze\b|\brc\d\b",
    re.I,
)

# ── Change intent / category features ────────────────────────────────────────
_TEST_CHANGE     = re.compile(
    r"\b(test|tests|testing|unittest|unit.test|pytest|coverage|fixture|mock|"
    r"e2e|integration.test|functional.test)\b",
    re.I,
)
_FIX_CHANGE      = re.compile(
    r"\b(fix|fixes|fixed|bugfix|bug.fix|hotfix|patch|issue|error|defect|"
    r"incorrect|wrong|broken|repair|resolve|resolves|workaround)\b",
    re.I,
)
_REFACTOR_CHANGE = re.compile(
    r"\b(refactor|refactoring|cleanup|clean.up|reorganize|restructure|"
    r"simplify|extract|rename|move|rework|tidy|readability|dedup)\b",
    re.I,
)
_FEAT_CHANGE     = re.compile(
    r"\b(add|adds|added|implement|implements|new|feature|support|introduce|"
    r"enable|expose|allow|provide|create)\b",
    re.I,
)
_DOCS_CHANGE     = re.compile(
    r"\b(doc|docs|documentation|readme|changelog|docstring|comment|"
    r"typo|spelling|grammar|wording)\b",
    re.I,
)
_CONV_PREFIX     = re.compile(
    r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)"
    r"(\(.+?\))?(!)?:",
    re.I,
)


class SubmissionFeatureBuilder:
    """Build the feature matrix X and target vector y from a split DataFrame.

    Parameters
    ----------
    numeric_cols:
        Names of numeric columns to include directly (after imputation + scaling).
    cat_cols:
        Names of low-cardinality categorical columns (one-hot encoded).
    owner_hash_n:
        Number of hash buckets for the high-cardinality owner_id feature.
    """

    NUMERIC_DEFAULTS: list[str] = [
        # Code churn
        "insertions", "deletions", "churn_total",
        # Submission metadata
        "hashtags_count", "has_topic",
        "created_dayofweek", "created_hour", "created_month",
        "days_to_nearest_release", "has_release_keyword", "is_blueprint",
        # Author submission history (per-project)
        "prior_submission_count", "prior_merged_count", "prior_abandoned_count",
        "prior_hp_count", "prior_hp_rate", "prior_cr_plus2_count",
        "days_since_first_contribution",
        "prior_90d_submissions", "prior_365d_submissions",
        # Author quality ratios
        "prior_merged_ratio", "prior_abandoned_ratio", "prior_cr_plus2_rate",
        "reputation_score", "log_prior_submission_count", "log_prior_cr_plus2_count",
        # Core developer status
        "is_core_developer",
        # Recent activity windows
        "prior_90d_merged", "prior_365d_merged",
        "prior_90d_submission_velocity", "prior_submission_recency_ratio",
        # Per-change quality proxies
        "prior_avg_patchset_count", "prior_avg_churn",
        "prior_cr_minus2_count", "prior_avg_review_latency_h",
        "prior_submission_days_std",
        # Cross-project (global) features
        "global_prior_submissions", "global_prior_merged",
        "global_prior_hp_count", "global_prior_unique_projects",
        # HP+ reviewer history
        "prior_hp_assigned_count", "log_prior_hp_assigned_count",
        "churn_x_is_core",
        # Text features (computed if subject available)
        "subject_len", "subject_word_count",
        # Subject-derived CI-blocker signals (from developer interviews)
        "is_stable_backport",         # stable/* branch or [stable] prefix — 1.80x HP+ lift
        "is_revert",                  # "Revert ..." commits — 1.66x HP+ lift
        "release_or_milestone_keyword",  # milestone/freeze/RC in subject — 1.54x lift
        "maintainer_attention_keyword",  # urgent/regression/data-loss in subject
        "is_ci_infra",                # CI/tox/zuul-related changes
        "is_security_or_cve",         # CVE or security fix in subject
        # Previously computed but unused taxonomy features
        "is_db_refactoring", "is_ovn_related", "is_rbac_policy",
        "has_bracket_prefix", "is_driver_work", "vendor_mentioned",
        # Change intent / category (from commit subject NLP)
        "is_test_commit", "is_fix_commit", "is_refactor_commit",
        "is_feat_commit", "is_docs_commit", "has_conv_prefix",
        # Author HP+ assignment history rate
        "prior_hp_assigned_rate",
    ]
    CAT_DEFAULTS: list[str] = ["branch_type", "size_label", "topic_category"]

    def __init__(
        self,
        numeric_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
        owner_hash_n: int = 512,
    ) -> None:
        self.numeric_cols  = numeric_cols or self.NUMERIC_DEFAULTS
        self.cat_cols      = cat_cols or self.CAT_DEFAULTS
        self.owner_hash_n  = owner_hash_n
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive text-based features from the 'subject' column in-place."""
        df = df.copy()
        subjects = df.get("subject", pd.Series([""] * len(df), index=df.index))
        subjects = subjects.fillna("").astype(str)

        df["subject_len"]         = subjects.str.len()
        df["subject_word_count"]  = subjects.str.split().apply(len)
        df["has_release_keyword"] = subjects.apply(lambda s: int(bool(_RELEASE_KEYWORDS.search(s))))
        df["is_blueprint"]        = subjects.apply(lambda s: int(bool(_BLUEPRINT_PREFIX.search(s))))

        # Taxonomy-derived binary features
        df["is_stable_backport"]  = subjects.apply(lambda s: int(bool(_STABLE_BACKPORT.search(s))))
        df["is_revert"]           = subjects.apply(lambda s: int(bool(_REVERT.search(s))))
        df["is_db_refactoring"]   = subjects.apply(lambda s: int(bool(_DB_REFACTOR.search(s))))
        df["is_ovn_related"]      = subjects.apply(lambda s: int(bool(_OVN_RELATED.search(s))))
        df["is_rbac_policy"]      = subjects.apply(lambda s: int(bool(_RBAC_POLICY.search(s))))
        df["is_ci_infra"]         = subjects.apply(lambda s: int(bool(_CI_INFRA.search(s))))
        df["has_bracket_prefix"]  = subjects.apply(lambda s: int(bool(_BRACKET_PREFIX.search(s))))
        df["is_driver_work"]      = subjects.apply(lambda s: int(bool(_DRIVER_WORK.search(s))))
        df["vendor_mentioned"]    = subjects.apply(lambda s: int(bool(_VENDOR_NAMES.search(s))))
        df["is_security_or_cve"]  = subjects.apply(lambda s: int(bool(_SECURITY_CVE.search(s))))
        df["maintainer_attention_keyword"] = subjects.apply(lambda s: int(bool(_MAINTAINER_ATTN.search(s))))
        df["release_or_milestone_keyword"] = subjects.apply(lambda s: int(bool(_MILESTONE_KW.search(s))))

        # Change intent / category features
        df["is_test_commit"]     = subjects.apply(lambda s: int(bool(_TEST_CHANGE.search(s))))
        df["is_fix_commit"]      = subjects.apply(lambda s: int(bool(_FIX_CHANGE.search(s))))
        df["is_refactor_commit"] = subjects.apply(lambda s: int(bool(_REFACTOR_CHANGE.search(s))))
        df["is_feat_commit"]     = subjects.apply(lambda s: int(bool(_FEAT_CHANGE.search(s))))
        df["is_docs_commit"]     = subjects.apply(lambda s: int(bool(_DOCS_CHANGE.search(s))))
        df["has_conv_prefix"]    = subjects.apply(lambda s: int(bool(_CONV_PREFIX.match(s))))

        # Also enrich branch-based stable detection
        if "branch" in df.columns:
            branch = df["branch"].fillna("").astype(str)
            df["is_stable_backport"] = (
                df["is_stable_backport"]
                | branch.str.startswith("stable/").astype(int)
            ).astype(int)

        return df

    def add_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count same-author HP+ changes and submissions in the last 7 days.

        Vectorised per-owner rolling window; uses only strictly past rows (no leakage).
        """
        df = df.copy()
        if "created_ts" not in df.columns or "owner_id" not in df.columns:
            df["n_hp_same_author_last_7d"]         = 0
            df["n_submissions_same_author_last_7d"] = 0
            return df

        df["_ts"] = pd.to_datetime(df["created_ts"], utc=True, errors="coerce")
        has_hp = "hp_positive" in df.columns

        n_sub_out = np.zeros(len(df), dtype=int)
        n_hp_out  = np.zeros(len(df), dtype=int)
        window_ns = pd.Timedelta(days=7).value  # nanoseconds

        for owner, grp in df.groupby("owner_id", sort=False):
            grp_sorted = grp.dropna(subset=["_ts"]).sort_values("_ts")
            if grp_sorted.empty:
                continue
            ts_arr  = grp_sorted["_ts"].values.astype(np.int64)
            hp_arr  = grp_sorted["hp_positive"].values.astype(int) if has_hp else np.zeros(len(grp_sorted), int)
            pos_map = {orig_idx: pos for pos, orig_idx in enumerate(grp_sorted.index)}

            for pos, (orig_idx, row_ts) in enumerate(zip(grp_sorted.index, ts_arr)):
                lo = np.searchsorted(ts_arr, row_ts - window_ns, side="left")
                hi = np.searchsorted(ts_arr, row_ts, side="left")   # strictly < current
                n_sub_out[df.index.get_loc(orig_idx)] = hi - lo
                n_hp_out[df.index.get_loc(orig_idx)]  = int(hp_arr[lo:hi].sum())

        df["n_hp_same_author_last_7d"]         = n_hp_out
        df["n_submissions_same_author_last_7d"] = n_sub_out
        df.drop(columns=["_ts"], inplace=True)
        return df

    def add_release_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add days_to_nearest_release if not already present."""
        df = df.copy()
        if "days_to_nearest_release" not in df.columns and "created_ts" in df.columns:
            df["days_to_nearest_release"] = df["created_ts"].apply(
                lambda ts: days_to_nearest_release(ts) if pd.notna(ts) else -1
            )
        return df

    def fit_transform(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit on *train* and return (X_train, y_train)."""
        train = self._enrich(train)
        self._pipeline = self._build_pipeline(train)
        X = self._pipeline.fit_transform(train)
        y = train["hp_positive"].to_numpy(dtype=int)
        log.info("fit_transform: X.shape=%s, HP+ rate=%.1f%%",
                 X.shape, 100 * y.mean())
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to a held-out set."""
        if self._pipeline is None:
            raise RuntimeError("Call fit_transform first.")
        df = self._enrich(df)
        return self._pipeline.transform(df)

    def get_feature_names(self) -> list[str]:
        """Return feature names after fitting (best-effort)."""
        if self._pipeline is None:
            return []
        try:
            return list(self._pipeline.named_steps["prep"].get_feature_names_out())
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_text_features(df)
        df = self.add_release_proximity(df)
        if "subject" in df.columns:
            df["subject"] = df["subject"].fillna("").astype(str)
        return df

    def _build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        present_num = [c for c in self.numeric_cols if c in df.columns]
        present_cat = [c for c in self.cat_cols if c in df.columns]

        transformers: list[Any] = [
            ("num", Pipeline([
                ("imp",   SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), present_num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), present_cat),
        ]

        # Owner ID: high-cardinality → hashing trick
        if "owner_id" in df.columns:
            transformers.append(
                ("owner", _OwnerHasher(n_features=self.owner_hash_n), "owner_id")
            )

        # TF-IDF on commit subjects (fit on train only → no leakage)
        if "subject" in df.columns and df["subject"].str.len().sum() > 0:
            n_components = min(30, max(2, len(df) // 10 - 1))
            transformers.append(
                ("tfidf_svd", Pipeline([
                    ("tfidf", TfidfVectorizer(
                        max_features=300, min_df=1, sublinear_tf=True,
                        strip_accents="unicode", analyzer="word",
                    )),
                    ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
                ]), "subject")
            )

        return Pipeline([
            ("prep", ColumnTransformer(transformers, remainder="drop")),
        ])


# ---------------------------------------------------------------------------
# Custom transformer for owner hashing
# ---------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class _OwnerHasher(BaseEstimator, TransformerMixin):
    """Hash owner IDs into a fixed-size binary vector."""

    def __init__(self, n_features: int = 512) -> None:
        self.n_features = n_features
        self._hasher = FeatureHasher(n_features=n_features, input_type="string")

    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        return self._hasher.transform(
            [{"owner_" + str(v): 1} for v in X]
        ).toarray()

    def get_feature_names_out(self, input_features=None):
        return [f"owner_hash_{i}" for i in range(self.n_features)]
