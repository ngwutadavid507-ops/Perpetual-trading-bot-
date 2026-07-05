"""
AI-powered signal filter using Claude API.
After the scanner collects all qualifying signals each cycle,
this module sends them to Claude which:
1. Evaluates each signal's quality in context
2. Selects only the best 1-2 signals per scan
3. Enforces the daily limit of 15 signals
4. Returns reasoning for why each signal was selected or rejected
"""

import json
import logging
import os
from datetime import datetime, date

import requests

logger = logging.getLogger(__name__)

DAILY_LOG_FILE = "/tmp/daily_signals.json"
MAX_SIGNALS_PER_DAY = 15
MAX_SIGNALS_PER_SCAN = 2


def _load_daily_log() -> dict:
    try:
        if os.path.exists(DAILY_LOG_FILE):
            with open(DAILY_LOG_FILE, "r") as f:
                data = json.load(f)
                if data.get("date") == str(date.today()):
                    return data
    except Exception:
        pass
    return {"date": str(date.today()), "count": 0, "signals": []}


def _save_daily_log(log: dict):
    try:
        with open(DAILY_LOG_FILE, "w") as f:
            json.dump(log, f)
    except Exception:
        pass


def get_daily_count() -> int:
    return _load_daily_log()["count"]


def increment_daily_count(signal_summary: str):
    log = _load_daily_log()
    log["count"] += 1
    log["signals"].append({
        "time": datetime.utcnow().isoformat(),
        "summary": signal_summary,
    })
    _save_daily_log(log)


def remaining_today() -> int:
    return max(0, MAX_SIGNALS_PER_DAY - get_daily_count())


def ai_select_signals(candidates: list) -> list:
    """
    Takes a list of Signal objects, sends them to Claude for evaluation,
    and returns only the top signals worth sending.

    Returns a filtered list of (Signal, df) tuples.
    """
    if not candidates:
        return []

    daily_remaining = remaining_today()
    if daily_remaining <= 0:
        logger.info(f"[ai_filter] Daily limit reached ({MAX_SIGNALS_PER_DAY})")
        return []

    # Cap at per-scan limit and daily remaining
    max_to_send = min(MAX_SIGNALS_PER_SCAN, daily_remaining)

    # If only one candidate just return it directly — no need for AI
    if len(candidates) <= max_to_send:
        for sig, _ in candidates:
            increment_daily_count(f"{sig.symbol} {sig.direction} {sig.confidence}%")
        return candidates

    # Build signal descriptions for Claude to evaluate
    signal_descriptions = []
    for i, (sig, _) in enumerate(candidates):
        signal_descriptions.append({
            "index": i,
            "symbol": sig.symbol.replace(":USDT", "").replace("/USDT", ""),
            "exchange": sig.exchange,
            "direction": sig.direction,
            "confidence": sig.confidence,
            "leverage": sig.leverage,
            "sl_pct": sig.sl_pct,
            "tp1_pct": sig.tp1_pct,
            "tp2_pct": sig.tp2_pct,
            "tp3_pct": sig.tp3_pct,
            "reasons": sig.reasons,
        })

    prompt = f"""You are a professional crypto futures trading signal evaluator.

I have {len(candidates)} qualifying signals from a scan. I can only send {max_to_send} signal(s) this cycle. I have {daily_remaining} signals remaining today out of {MAX_SIGNALS_PER_DAY} maximum.

Here are the candidates:
{json.dumps(signal_descriptions, indent=2)}

Evaluate each signal and select the best {max_to_send} based on:
1. Confidence score (higher is better)
2. Quality and consistency of reasons (more confluence = better)
3. Risk/reward profile (tighter SL % with good TP % = better)
4. Avoid selecting two signals in the same direction if possible
5. Prefer signals with more reasons listed

Respond ONLY with a JSON object in this exact format, no other text:
{{
  "selected": [0, 2],
  "reasoning": "Brief explanation of why these were chosen over others"
}}

The "selected" array should contain the index numbers of the signals you choose."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        raw = data["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        selected_indices = result.get("selected", [])
        reasoning = result.get("reasoning", "")
        logger.info(f"[ai_filter] Claude selected indices {selected_indices}: {reasoning}")

        selected = []
        for idx in selected_indices:
            if 0 <= idx < len(candidates):
                sig, df = candidates[idx]
                increment_daily_count(
                    f"{sig.symbol} {sig.direction} {sig.confidence}%"
                )
                selected.append((sig, df))

        return selected

    except Exception as e:
        logger.error(f"[ai_filter] Claude API call failed: {e}")
        # Fallback — sort by confidence and return top signals
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x[0].confidence,
            reverse=True
        )[:max_to_send]
        for sig, _ in sorted_candidates:
            increment_daily_count(
                f"{sig.symbol} {sig.direction} {sig.confidence}%"
            )
        return sorted_candidates
