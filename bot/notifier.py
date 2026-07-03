import logging
from telegram import Bot
from telegram.constants import ParseMode

from bot.signal_engine import Signal

logger = logging.getLogger(__name__)


def format_signal_message(sig: Signal) -> str:
    arrow = "🟢 LONG" if sig.direction == "long" else "🔴 SHORT"
    reasons_block = "\n".join(f"• {r}" for r in sig.reasons)

    if sig.direction == "long":
        sl_note = "❌ SL hit = exit full position"
        tp1_note = "⚡ TP1 = close 50% of position, move SL to entry (risk free)"
        tp2_note = "🎯 TP2 = close remaining 50%"
    else:
        sl_note = "❌ SL hit = exit full position"
        tp1_note = "⚡ TP1 = close 50% of position, move SL to entry (risk free)"
        tp2_note = "🎯 TP2 = close remaining 50%"

    return (
        f"*{arrow} — {sig.symbol}* ({sig.exchange})\n\n"
        f"Confidence: *{sig.confidence}%*\n"
        f"Leverage: *{sig.leverage}x*\n\n"
        f"Entry: `{sig.entry}`\n"
        f"Stop Loss: `{sig.stop_loss}`\n"
        f"TP1: `{sig.take_profit1}` (R:R 1:{sig.risk_reward1})\n"
        f"TP2: `{sig.take_profit2}` (R:R 1:{sig.risk_reward2})\n\n"
        f"📋 *Trade Plan:*\n"
        f"{tp1_note}\n"
        f"{tp2_note}\n"
        f"{sl_note}\n\n"
        f"Reasons:\n{reasons_block}\n\n"
        f"_Not financial advice. Confirm on your own chart before entering._"
    )


async def send_signal(bot_token: str, chat_id: str, sig: Signal):
    bot = Bot(token=bot_token)
    text = format_signal_message(sig)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


async def send_result(bot_token: str, chat_id: str, symbol: str, direction: str, result: str, pnl_r: float):
    bot = Bot(token=bot_token)
    if result == "tp1":
        emoji = "⚡"
        title = "TP1 HIT — Move SL to entry now"
    elif result == "tp2":
        emoji = "🎯"
        title = "TP2 HIT — Full target reached"
    elif result == "sl":
        emoji = "❌"
        title = "SL HIT — Loss taken"
    else:
        emoji = "⏱"
        title = "Signal expired"

    direction_label = "LONG" if direction == "long" else "SHORT"
    text = (
        f"{emoji} *{title}*\n\n"
        f"Signal: {direction_label} — {symbol}\n"
        f"Result: *{'+' if pnl_r > 0 else ''}{pnl_r}R*"
    )
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Failed to send result message: {e}")
