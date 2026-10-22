"""Generate structured natural-language explanations via NVIDIA NIM (Nemotron)."""

from __future__ import annotations

import os
from openai import OpenAI

_client: OpenAI | None = None

# The original model (nvidia/llama-3.3-nemotron-super-49b-v1) was retired by
# NVIDIA on 2026-08-26. Override with NVIDIA_MODEL; set NVIDIA_THINKING=1 for
# reasoning mode (slower, e.g. with nvidia/nemotron-3-ultra-550b-a55b).
MODEL_ID = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
THINKING = os.environ.get("NVIDIA_THINKING", "0").lower() in ("1", "true", "yes")


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in environment")
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
    return _client


_SYSTEM_PROMPT = """\
You are a senior OpenStack code reviewer. A machine-learning model has classified a Gerrit \
change as either HP+ (high-priority, needs urgent review) or Normal (standard review pace). \
Your task is to write a coherent technical explanation where EVERY section leads to the same \
conclusion as the verdict — the narrative and the verdict must never contradict each other.

Return EXACTLY this structure — no extra text before or after:

**Verdict**: [One sentence: what this change does technically AND the single decisive reason \
it is HP+ or Normal — both must be in the same sentence]

**Key Signals**:
- [The strongest technical argument FOR the verdict — be specific about subsystem or risk]
- [Second supporting argument — scope, blocking dependency, stability risk, or lack thereof]
- [Third argument — drawn from commit description, changed files, or reviewer comments]

**Why It Matters**: [If HP+: one sentence on the concrete consequence of delay — what breaks, \
who is blocked, or what regression occurs. If Normal: one sentence explaining why, despite any \
complexity or broad scope, this does NOT require urgent attention — e.g. it is additive, \
opt-in, behind a version gate, or has no effect on existing deployments]

**Suggestion**: [One actionable sentence for the reviewer team that matches the urgency level]

Critical rules:
- ALL four sections must point toward the SAME verdict. If the change touches many files but \
is still Normal, explain why the scope does not create urgency. If it is HP+, every bullet \
must reinforce urgency. Never describe something as risky or blocking in Key Signals and \
then call it Normal in the Verdict — or vice versa.
- Focus on the TECHNICAL nature of the change: what it fixes, what behavior it changes, \
what it risks, whether it is backward-compatible, whether it blocks other work.
- Do NOT mention model feature names, weights, or statistics.
- Do NOT mention the author's history, reputation, or past submissions.
- Reference reviewer comments and changed files when available to ground the explanation.
- Keep each bullet under 25 words. Be specific, not generic."""


def explain(
    subject: str,
    project: str,
    probability: float,
    verdict: str,
    top_features: list[dict],
    matched_category: str,
    hp_rate: float,
    changed_files: list[str] | None = None,
    recent_comments: list[str] | None = None,
    commit_body: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Call NIM to produce a structured 4-section technical explanation.

    Returns a markdown string, or a fallback message on error.
    top_features is accepted for API compatibility but intentionally not passed
    to the LLM — the explanation should reason from the change content, not ML features.
    """
    files_block = ""
    if changed_files:
        files_block = "Changed files:\n" + "\n".join(
            f"  {p}" for p in changed_files
        ) + "\n\n"

    body_block = ""
    if commit_body and commit_body.strip():
        body_block = f"Commit description:\n  {commit_body[:600]}\n\n"

    comments_block = ""
    if recent_comments:
        comments_block = "Recent reviewer comments:\n" + "\n".join(
            f"  [{i+1}] {c}" for i, c in enumerate(recent_comments)
        ) + "\n\n"

    user_msg = (
        f"Project: {project}  |  Category: {matched_category}\n"
        f"Change subject: {subject!r}\n"
        f"ML model verdict: {verdict} ({probability:.1%} HP+ probability)\n\n"
        f"{body_block}"
        f"{files_block}"
        f"{comments_block}"
        "Based on the change content above, produce the structured technical explanation."
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            # Reasoning tokens count against max_tokens, so thinking needs more room
            max_tokens=3000 if THINKING else 420,
            temperature=0.2,
            timeout=max(timeout, 90.0) if THINKING else timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": THINKING}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return "[LLM explanation unavailable: empty response from model]"
        return content
    except Exception as exc:  # noqa: BLE001
        return f"[LLM explanation unavailable: {exc}]"
