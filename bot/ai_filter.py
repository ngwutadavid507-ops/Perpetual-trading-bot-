"""
AI-powered signal filter using Claude API.
Daily limit and signal log stored in Redis — persists across restarts.
"""

import json
import logging
import os
from datetime import date

import requests as http_requests

from bot.redis_client import redis_get, redis_set, redis_incr, redis_expire

logger = logging.getLogger(__name__)

MAX_SIGNALS_PER_DAY = 15
MAX_SIGNALS_PER_SCAN = 2

DAILY_COUNT_KEY = f"daily_count:{date.today()}"
DAILY_LOG_KEY = f"daily_log:{date.today()}"


def _today_count_key() -> str:
    return f"daily_count:{date.today()}"


def _today_log_key() -> str:
    return f"daily_log:{date.today()}"


def get_daily_count() -> int:
    val = redis_get(_today_count_key())
    return int(val) if val else 0


def remaining_today() -> int:
    return max(0, MAX_SIGNALS_PER_DAY - get_daily_count())


def increment_daily_count(signal_summary: str):
    key = _today_count_key()
    count = redis_incr(key)
    if count == 1:
        # First increment — set expiry to 48 hours so it auto-cleans
        redis_expire(key, 48 * 3600)

    # Append to log
    log_key = _today_log_key()
    existing = redis_get(log_key)
    logs = json.loads(existing) if existing else []
    logs.append(signal_summary)
    redis_set(log_key, json.dumps(logs), ex=48 * 3600)


def ai_select_signals(candidates: list) -> list:
    if not candidates:
        return []

    daily_remaining = remaining_today()
    if daily_remaining <= 0:
        logger.info(f"[ai_filter] Daily limit reached ({MAX_SIGNALS_PER_DAY})")
        return []

    max_to_send = min(MAX_SIGNALS_PER_SCAN, daily_remaining)

    if len(candidates) <= max_to_send:
        for sig, _ in candidates:
            increment_daily_count(
                f"{sig.symbol} {sig.direction} {sig.confidence}%"
            )
        return candidates

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

I have {len(candidates)} qualifying signals. I can only send {max_to_send} signal(s) this cycle. I have {daily_remaining} signals remaining today out of {MAX_SIGNALS_PER_DAY} maximum.

Candidates:
{json.dumps(signal_descriptions, indent=2)}

Select the best {max_to_send} based on:
1. Confidence score (higher is better)
2. Quality and consistency of reasons (more confluence = better)
3. Risk/reward profile (tighter SL % with good TP % = better)
4. Avoid two signals in the same direction if possible
5. Prefer signals with more reasons listed

Respond ONLY with JSON, no other text:
{{
  "selected": [0, 2],
  "reasoning": "Brief explanation"
}}"""

    try:
        response = http_requests.post(
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
        logger.info(f"[ai_filter] selected {selected_indices}: {reasoning}")

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
        logger.error(f"[ai_filter] Claude API failed: {e}")
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
