"""
Daily summary generator — Redis backed.
Tracks all signal results throughout the day and sends
a professional summary to Telegram at midnight UTC.
Includes win rate, ROI, grade breakdown and strategy performance.
"""

import json
import logging
from datetime import datetime, date

from bot.redis_client import redis_get, redis_set

logger = logging.getLogger(__name__)


def _today_key() -> str:
    return f"summary:v2:{date.today()}"


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
        "continuation_wins": 0,
        "continuation_losses": 0,
        "reversal_wins": 0,
        "reversal_losses": 0,
        "grades": {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0},
        "results": [],
    }


def _save_summary(data: dict):
    redis_set(_today_key(), json.dumps(data), ex=48 * 3600)


def record_signal_sent(
    symbol: str,
    direction: str,
    confidence: float,
    strategy_type: str,
    leverage: int,
):
    data = _load_summary()
    data["signals_sent"] += 1
    data["results"].append({
        "symbol": symbol.replace(":USDT", "").replace("/USDT", ""),
        "direction": direction,
        "confidence": confidence,
        "strategy_type": strategy_type,
        "leverage": leverage,
        "result": "open",
        "pnl_r": 0.0,
        "roi_pct": 0.0,
        "grade": "",
        "time": datetime.utcnow().strftime("%H:%M"),
    })
    _save_summary(data)


def record_result(
    symbol: str,
    direction: str,
    result: str,
    pnl_r: float,
    leverage: int = 1,
    strategy_type: str = "unknown",
    grade: str = "",
):
    data = _load_summary()
    symbol_clean = symbol.replace(":USDT", "").replace("/USDT", "")

    # Calculate approximate ROI
    sl_pct = abs(pnl_r) / max(abs(pnl_r), 0.01) * 1.0
    roi_pct = round(pnl_r * leverage, 2)

    if result == "tp1":
        data["tp1_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
        if strategy_type == "continuation":
            data["continuation_wins"] += 1
        elif strategy_type == "reversal":
            data["reversal_wins"] += 1
    elif result == "tp2":
        data["tp2_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
        if strategy_type == "continuation":
            data["continuation_wins"] += 1
        elif strategy_type == "reversal":
            data["reversal_wins"] += 1
    elif result == "tp3":
        data["tp3_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
        if strategy_type == "continuation":
            data["continuation_wins"] += 1
        elif strategy_type == "reversal":
            data["reversal_wins"] += 1
    elif result == "sl":
        data["sl_hits"] += 1
        data["total_r"] = round(data["total_r"] + pnl_r, 2)
        if strategy_type == "continuation":
            data["continuation_losses"] += 1
        elif strategy_type == "reversal":
            data["reversal_losses"] += 1

    if grade and grade in data["grades"]:
        data["grades"][grade] += 1

    # Update matching open result
    for r in data["results"]:
        if r["symbol"] == symbol_clean and r["result"] == "open":
            r["result"] = result
            r["pnl_r"] = pnl_r
            r["roi_pct"] = roi_pct
            r["grade"] = grade
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
    grades = data["grades"]

    cont_wins = data.get("continuation_wins", 0)
    cont_losses = data.get("continuation_losses", 0)
    rev_wins = data.get("reversal_wins", 0)
    rev_losses = data.get("reversal_losses", 0)

    closed = tp1 + tp2 + tp3 + sl
    wins = tp1 + tp2 + tp3
    win_rate = round(wins / closed * 100) if closed > 0 else 0
    total_r_str = f"+{total_r}R" if total_r >= 0 else f"{total_r}R"

    # Results list
    results_block = ""
    for r in results:
        direction_emoji = "🟢" if r["direction"] == "long" else "🔴"
        strategy_icon = "📈" if r.get("strategy_type") == "continuation" else "🔄"

        if r["result"] == "tp1":
            res_str = f"⚡ +{r['pnl_r']}R | +{r['roi_pct']}% (TP1)"
        elif r["result"] == "tp2":
            res_str = f"🎯 +{r['pnl_r']}R | +{r['roi_pct']}% (TP2)"
        elif r["result"] == "tp3":
            res_str = f"🏆 +{r['pnl_r']}R | +{r['roi_pct']}% (TP3)"
        elif r["result"] == "sl":
            res_str = f"❌ {r['pnl_r']}R | {r['roi_pct']}% (SL)"
        else:
            res_str = "⏳ Open"

        grade_str = f" [{r['grade']}]" if r.get("grade") else ""
        results_block += (
            f"{strategy_icon}{direction_emoji} {r['symbol']} "
            f"{r['confidence']}% {r['leverage']}x "
            f"| {res_str}{grade_str}\n"
        )

    # Strategy breakdown
    cont_total = cont_wins + cont_losses
    rev_total = rev_wins + rev_losses
    cont_wr = round(cont_wins / cont_total * 100) if cont_total > 0 else 0
    rev_wr = round(rev_wins / rev_total * 100) if rev_total > 0 else 0

    # Grade breakdown
    grade_str = (
        f"A+:{grades.get('A+', 0)} "
        f"A:{grades.get('A', 0)} "
        f"B:{grades.get('B', 0)} "
        f"C:{grades.get('C', 0)} "
        f"D:{grades.get('D', 0)}"
    )

    return (
        f"📊 *Phoenix Daily Summary — {today}*\n\n"
        f"Signals: *{signals_sent}* | Closed: *{closed}*\n"
        f"Win Rate: *{win_rate}%* ({wins}W / {sl}L)\n"
        f"Total R: *{total_r_str}*\n\n"
        f"📈 Continuation: {cont_wins}W/{cont_losses}L ({cont_wr}%)\n"
        f"🔄 Reversal: {rev_wins}W/{rev_losses}L ({rev_wr}%)\n\n"
        f"Grades: {grade_str}\n\n"
        f"*Results:*\n{results_block}\n"
        f"_Phoenix Signal Bot 🔥_"
    )
