"""
LLM-generated compliance risk summaries for flagged users.

For every HIGH/CRITICAL alert, I send the anomaly signals to an LLM and ask
it to write a short compliance narrative — what the pattern looks like, why
it's risky, and what the next step should be. This makes the alerts actually
actionable for a compliance officer rather than just a list of feature names.

Backend priority:
  1. ANTHROPIC_API_KEY set -> Claude Haiku (fast and cheap)
  2. GEMINI_API_KEY set    -> Gemini 2.0 Flash (fallback)
  3. Neither               -> template string (pipeline never breaks)

I only call the LLM for HIGH/CRITICAL to control cost. MEDIUM and LOW just
get the raw explanation string from the ensemble.
"""

import logging
import os

import pandas as pd

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.llm.risk_summary", "forexguard_llm.log")

# ──────────────────────────────────────────────────────────────────────────────
# LLM backend selector
# ──────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> str:
    """Return 'claude', 'gemini', or 'none'."""
    if os.getenv("ANTHROPIC_API_KEY", ""):
        return "claude"
    if os.getenv("GEMINI_API_KEY", ""):
        return "gemini"
    return "none"


def _call_claude(prompt: str, system: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=256, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_gemini(prompt: str, system: str, model: str = "gemini-2.0-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a compliance officer at a regulated forex brokerage. Your job is to
review anomaly detection alerts and write concise, professional risk summaries
for the compliance team. Each summary must:
1. Identify the most likely anomaly pattern (from the signal names provided)
2. Estimate the risk level and why
3. Suggest 1-2 specific next steps (e.g., KYC review, transaction hold, IP block)
Keep the response under 120 words. Use formal compliance language. No bullet points — prose only."""

USER_PROMPT_TEMPLATE = """\
User ID: {user_id}
Alert Tier: {alert_tier}
Composite Risk Score: {composite_score:.3f}
Isolation Forest Score: {if_score:.3f}
LSTM Reconstruction Score: {lstm_score:.3f}

Anomaly Signals Detected:
{explanation}

Write a risk summary for this user."""


# ──────────────────────────────────────────────────────────────────────────────
# Single-user summary
# ──────────────────────────────────────────────────────────────────────────────

def generate_risk_summary(
    user_id: str,
    alert_tier: str,
    composite_score: float,
    if_score: float,
    lstm_score: float,
    explanation: str,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """
    Generate a compliance risk narrative for one user.
    Uses Claude if ANTHROPIC_API_KEY is set, Gemini if GEMINI_API_KEY is set,
    otherwise returns a template string.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_id         = user_id,
        alert_tier      = alert_tier,
        composite_score = composite_score,
        if_score        = if_score,
        lstm_score      = lstm_score,
        explanation     = explanation,
    )

    backend = _detect_backend()
    try:
        if backend == "claude":
            log.debug("Using Claude for user %s", user_id)
            return _call_claude(user_prompt, SYSTEM_PROMPT, model)
        elif backend == "gemini":
            log.debug("Using Gemini for user %s", user_id)
            return _call_gemini(user_prompt, SYSTEM_PROMPT)
        else:
            return (
                f"[Demo] {user_id} flagged as {alert_tier} risk "
                f"(score={composite_score:.3f}). "
                f"Signals: {explanation[:200]}. "
                "Recommend compliance review and enhanced due diligence."
            )
    except Exception as exc:
        log.warning("LLM call failed for user %s (%s): %s", user_id, backend, exc)
        return f"[LLM summary unavailable: {exc}]"


# ──────────────────────────────────────────────────────────────────────────────
# Batch generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_batch_summaries(
    alerts_df: pd.DataFrame,
    max_users: int = 50,
    model: str = "claude-haiku-4-5-20251001",
) -> pd.DataFrame:
    """
    Generate LLM risk summaries for the top `max_users` HIGH/CRITICAL alerts.
    Uses Claude (primary) or Gemini (fallback) based on available API keys.

    Parameters
    ----------
    alerts_df  : DataFrame with columns composite_score, if_score, lstm_score,
                 alert_tier, explanation (index = user_id)
    max_users  : cap on API calls (cost control)
    model      : Claude model ID (only used when Claude is the backend)

    Returns
    -------
    DataFrame with user_id index and 'llm_summary' column
    """
    backend = _detect_backend()
    log.info("LLM backend: %s", backend)

    # Only process HIGH and CRITICAL users, sorted by score desc
    target = alerts_df[alerts_df["alert_tier"].isin(["CRITICAL", "HIGH"])]
    target = target.sort_values("composite_score", ascending=False).head(max_users)

    if backend == "none":
        log.warning("No API key set (ANTHROPIC_API_KEY / GEMINI_API_KEY) — using placeholders.")

    log.info("Generating summaries for %d users...", len(target))
    summaries = {}
    for i, (uid, row) in enumerate(target.iterrows(), 1):
        log.info("  [%d/%d] %s (tier=%s)...", i, len(target), uid, row["alert_tier"])
        summary = generate_risk_summary(
            user_id         = uid,
            alert_tier      = str(row.get("alert_tier", "HIGH")),
            composite_score = float(row.get("composite_score", 0.0)),
            if_score        = float(row.get("if_score", 0.0)),
            lstm_score      = float(row.get("lstm_score", 0.0)),
            explanation     = str(row.get("explanation", "")),
            model           = model,
        )
        summaries[uid] = summary
        log.info("    -> %s", summary[:80] + "..." if len(summary) > 80 else summary)

    result = pd.DataFrame.from_dict(
        {"user_id": list(summaries.keys()), "llm_summary": list(summaries.values())}
    ).set_index("user_id")

    log.info("LLM summary generation complete. %d summaries produced.", len(result))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI integration helper — single-user on-demand summary
# ──────────────────────────────────────────────────────────────────────────────

def get_risk_summary_for_user(
    user_id: str,
    alert_tier: str,
    composite_score: float,
    if_score: float,
    lstm_score: float,
    explanation: str,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Convenience wrapper for the FastAPI /alerts endpoint."""
    return generate_risk_summary(
        user_id, alert_tier, composite_score, if_score, lstm_score, explanation, model
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    alerts_df = pd.read_parquet(raw_dir / "alerts.parquet")
    log.info("Loaded %d alerts.", len(alerts_df))

    summaries = generate_batch_summaries(alerts_df, max_users=10)

    out_path = raw_dir / "llm_summaries.parquet"
    summaries.to_parquet(out_path)
    log.info("Saved LLM summaries -> %s", out_path)

    print("\n--- Sample LLM Summaries ---")
    for uid, row in summaries.head(5).iterrows():
        print(f"\n[{uid}]")
        print(row["llm_summary"])
