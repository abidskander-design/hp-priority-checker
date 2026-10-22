"""Flask backend for HP+ Priority Checker Chrome extension."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load NVIDIA_API_KEY from hp-priority-checker/.env, falling back to the
# research repo's .env (original development layout). Variables already set
# in the environment (e.g. by docker compose) take precedence.
_TOOL_ROOT = Path(__file__).parent.parent
load_dotenv(_TOOL_ROOT / ".env")
load_dotenv(_TOOL_ROOT.parent / "review-priority-predictor" / ".env")

# SubmissionFeatureBuilder (needed to unpickle the models) is bundled in
# backend/rp_predict/, which is importable because it sits next to this file.
from db import init_db, log_prediction, get_history, get_latency_stats, get_recent_prediction  # noqa: E402
from gerrit_client import fetch_change, extract_change_info              # noqa: E402
from feature_extractor import build_feature_row                          # noqa: E402
from predictor import predict, supported_projects                        # noqa: E402
from llm_explainer import explain                                        # noqa: E402

app = Flask(__name__)
CORS(app, origins=["https://review.opendev.org"])


@app.before_request
def _init():
    """Initialise DB on first request."""
    if not hasattr(app, "_db_ready"):
        init_db()
        app._db_ready = True


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict_route():
    """Predict HP+ for a Gerrit change.

    Body (JSON):
        change_id      (str, required)  — numeric id, hash, or project~branch~Id
        project        (str, required)  — e.g. "openstack/nova"
        threshold_type (str, optional)  — "gmean" or "f05"  [default: "gmean"]
        threshold_override (float, optional) — user-adjusted threshold from slider

    Returns (JSON):
        probability, verdict, threshold_used, threshold_type,
        top_features, train_auc, train_mcc_gm, hp_rate,
        top_reviewers, matched_category, llm_explanation,
        subject, latency_ms
    """
    body = request.get_json(force=True, silent=True) or {}

    change_id      = body.get("change_id", "").strip()
    project        = body.get("project", "").strip()
    threshold_type = body.get("threshold_type", "gmean")
    thr_override   = body.get("threshold_override")  # float or None

    if not change_id:
        return jsonify({"error": "change_id is required"}), 400
    if not project:
        return jsonify({"error": "project is required"}), 400
    if project not in supported_projects():
        return jsonify({
            "error": f"Unsupported project: {project!r}",
            "supported": supported_projects(),
        }), 400

    t0 = time.time()

    # 1. Fetch from Gerrit
    try:
        raw_change = fetch_change(change_id)
    except Exception as exc:
        return jsonify({"error": f"Gerrit fetch failed: {exc}"}), 502

    change_info = extract_change_info(raw_change)
    subject     = change_info.get("subject", "")

    # 2. Build feature row
    feature_row = build_feature_row(change_info)

    # 3. Run XGBoost
    try:
        result = predict(feature_row, project, threshold_type=threshold_type)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # Apply user-adjusted threshold if provided
    if thr_override is not None:
        try:
            thr_override = float(thr_override)
            result["threshold_used"] = round(thr_override, 4)
            result["verdict"] = "HP+" if result["probability"] >= thr_override else "Normal"
            result["threshold_type"] = "custom"
        except (TypeError, ValueError):
            pass

    # 4. LLM explanation with enriched context
    llm_explanation = explain(
        subject          = subject,
        project          = project,
        probability      = result["probability"],
        verdict          = result["verdict"],
        top_features     = result["top_features"],
        matched_category = result["matched_category"],
        hp_rate          = result["hp_rate"],
        changed_files    = change_info.get("changed_files"),
        recent_comments  = change_info.get("recent_comments"),
        commit_body      = change_info.get("commit_body"),
    )

    latency_ms = round((time.time() - t0) * 1000, 1)

    # 5. Persist to SQLite
    try:
        log_prediction(
            change_id        = change_id,
            project          = project,
            subject          = subject,
            probability      = result["probability"],
            verdict          = result["verdict"],
            threshold_used   = result["threshold_used"],
            threshold_type   = result["threshold_type"],
            mcc_train        = result["train_mcc_gm"],
            auc_train        = result["train_auc"],
            matched_category = result["matched_category"],
            top_signals      = json.dumps(result["top_features"][:5]),
            llm_explanation  = llm_explanation,
            latency_ms       = latency_ms,
        )
    except Exception:
        pass  # log failure is non-critical

    return jsonify({
        **result,
        "subject":         subject,
        "llm_explanation": llm_explanation,
        "latency_ms":      latency_ms,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict_batch
# ─────────────────────────────────────────────────────────────────────────────

BATCH_MAX     = 50   # hard cap per request
BATCH_CACHE_S = 300  # reuse a prediction that's less than 5 minutes old


@app.route("/predict_batch", methods=["POST"])
def predict_batch_route():
    """Predict HP+ for multiple changes at once (used by list-page badges).

    Body (JSON):
        {"changes": [{"change_id": "12345", "project": "openstack/nova"}, ...]}

    Returns a JSON array.  Items whose project is unsupported or whose Gerrit
    fetch fails are silently omitted — callers must tolerate a shorter list.
    LLM explanation is skipped entirely for batch predictions.
    """
    body = request.get_json(force=True, silent=True) or {}
    changes = body.get("changes", [])[:BATCH_MAX]

    results = []
    for item in changes:
        change_id = str(item.get("change_id", "")).strip()
        project   = str(item.get("project", "")).strip()

        if not change_id or not project:
            continue
        if project not in supported_projects():
            continue

        # Fast path: reuse a recent prediction from the log
        cached = get_recent_prediction(change_id, project, max_age_seconds=BATCH_CACHE_S)
        if cached:
            results.append({
                "change_id":        change_id,
                "project":          project,
                "verdict":          cached["verdict"],
                "probability":      cached["probability"],
                "matched_category": cached["matched_category"],
            })
            continue

        # Slow path: fetch from Gerrit + run model
        try:
            raw_change  = fetch_change(change_id)
            change_info = extract_change_info(raw_change)
            feature_row = build_feature_row(change_info)
            result      = predict(feature_row, project)

            log_prediction(
                change_id        = change_id,
                project          = project,
                subject          = change_info.get("subject", ""),
                probability      = result["probability"],
                verdict          = result["verdict"],
                threshold_used   = result["threshold_used"],
                threshold_type   = result["threshold_type"],
                mcc_train        = result["train_mcc_gm"],
                auc_train        = result["train_auc"],
                matched_category = result["matched_category"],
                top_signals      = json.dumps(result["top_features"][:5]),
                llm_explanation  = "",   # skipped for batch
                latency_ms       = 0.0,
            )

            results.append({
                "change_id":        change_id,
                "project":          project,
                "verdict":          result["verdict"],
                "probability":      result["probability"],
                "matched_category": result["matched_category"],
            })
        except Exception:
            continue  # silently skip failed items

    return jsonify(results)


# ─────────────────────────────────────────────────────────────────────────────
# GET /history
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def history_route():
    limit = int(request.args.get("limit", 50))
    rows  = get_history(limit=limit)
    return jsonify(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GET /latency
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/latency", methods=["GET"])
def latency_route():
    stats = get_latency_stats()
    return jsonify(stats)


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/projects", methods=["GET"])
def projects_route():
    return jsonify({"projects": supported_projects()})


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_route():
    return jsonify({"status": "ok", "projects": len(supported_projects())})


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("HP_CHECKER_PORT", 5487))
    host = os.environ.get("HP_CHECKER_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
