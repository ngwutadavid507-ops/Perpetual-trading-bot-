"""
Daily summary generator — Redis backed, survives Railway restarts.
Tracks all signal results throughout the day and sends
a professional summary to the Telegram channel at midnight UTC.
"""

import json
import logging
from datetime import datetime, date

from bot.redis_client import redis_get, redis_set

logger = logging.getLogger(__name__)


def _today_key() -> str:
    return f"summary:{date.today()}"


def _load_summary() -> dict:
    raw = redis_get(_today_key())
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {
        "date": str(date.today()),
        "signals_sent": 0,
        "tp1_hits": 0,
        "tp2_hits": 0,
        "tp3_hits": 0,
        "sl_hits": 0,
        "total_r": 0.0,
        "results": [],
    }


def _save_summary(data: dict):
    redis_set(_today_key(), json.dumps(data), ex=48 * 3600)


def record_signal_sent(symbol: str, direction: str, confidence: float):
    data = _load_summary()
    data["signals_sent"] += 1
    data["results"].append({
        "symbol": symbol.replace(":USDT", "").replace("/USDT", ""),
        "direction": direction,
        "confidence": confidence,
        "result": "open",
        "pnl_r": 0.0,
        "time": datetime.utcnow().strftime("%H:%M"),
    })
    _save_summary(data)


def record_result(symbol: str, direction: str, result: str, pnl_r: float):
    data = _load_summary()
    symbol_clean = symbol.replace(":USDT", "").replace("/USDT", "")

    if result == "tp1":
        data["tp1_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
    elif result == "tp2":
        data["tp2_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
    elif result == "tp3":
        data["tp3_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
    elif result == "sl":
        data["sl_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)

    for r in data["results"]:
        if r["symbol"] == symbol_clean and r["result"] == "open":
            r["result"] = result
            r["pnl_r"] = pnl_r
            break

    _save_summary(data)


def format_daily_summary() -> str:
    data = _load_summary()
    today = data["date"]

    signals_sent = data["signals_sent"]
    tp1 = data["tp1_hits"]
    tp2 = data["tp2_hits"]
    tp3 = data["tp3_hits"]
    sl = data["sl_hits"]
    total_r = data["total_r"]
    results = data["results"]

    closed = tp1 + tp2 + tp3 + sl
    wins = tp1 + tp2 + tp3
    win_rate = round(wins / closed * 100) if closed > 0 else 0
    total_r_str = f"+{total_r}R" if total_r >= 0 else f"{total_r}R"

    results_block = ""
    for r in results:
        direction_emoji = "🟢" if r["direction"] == "long" else "🔴"
        if r["result"] == "tp1":
            res_emoji = "⚡"
            res_str = f"+{r['pnl_r']}R (TP1)"
        elif r["result"] == "tp2":
            res_emoji = "🎯"
            res_str = f"+{r['pnl_r']}R (TP2)"
        elif r["result"] == "tp3":
            res_emoji = "🏆"
            res_str = f"+{r['pnl_r']}R (TP3)"
        elif r["result"] == "sl":
            res_emoji = "❌"
            res_str = f"{r['pnl_r']}R (SL)"
        else:
            res_emoji = "⏳"
            res_str = "Open"

        results_block += (
            f"{res_emoji} {direction_emoji} {r['symbol']} "
            f"| {r['confidence']}% | {res_str}\n"
        )

    return (
        f"📊 *Phoenix Daily Summary — {today}*\n\n"
        f"Signals sent: *{signals_sent}*\n"
        f"Closed trades: *{closed}*\n"
        f"Win rate: *{win_rate}%* ({wins}W / {sl}L)\n\n"
        f"TP1 hits: *{tp1}*\n"
        f"TP2 hits: *{tp2}*\n"
        f"TP3 hits: *{tp3}*\n"
        f"SL hits: *{sl}*\n\n"
        f"Total R: *{total_r_str}*\n\n"
        f"*Results:*\n{results_block}\n"
        f"_Phoenix Signal Bot 🔥_"
    )
