import logging
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode

from bot.signal_engine import Signal

logger = logging.getLogger(__name__)


def format_signal_message(sig: Signal) -> str:
    arrow = "🟢 LONG" if sig.direction == "long" else "🔴 SHORT"
    symbol_clean = sig.symbol.replace(":USDT", "").replace("/USDT", "")

    if sig.direction == "long":
        sl_label = f"`{sig.stop_loss}` (-{sig.sl_pct}%)"
        tp1_label = f"`{sig.take_profit1}` (+{sig.tp1_pct}%)"
        tp2_label = f"`{sig.take_profit2}` (+{sig.tp2_pct}%)"
    else:
        sl_label = f"`{sig.stop_loss}` (+{sig.sl_pct}%)"
        tp1_label = f"`{sig.take_profit1}` (-{sig.tp1_pct}%)"
        tp2_label = f"`{sig.take_profit2}` (-{sig.tp2_pct}%)"

    reasons_block = "\n".join(f"• {r}" for r in sig.reasons)

    return (
        f"*{arrow} — {symbol_clean}* ({sig.exchange})\n"
        f"Confidence: *{sig.confidence}%* | Leverage: *{sig.leverage}x*\n\n"
        f"Entry:  `{sig.entry}`\n"
        f"TP1:    {tp1_label}\n"
        f"TP2:    {tp2_label}\n"
        f"SL:     {sl_label}\n\n"
        f"{reasons_block}\n\n"
        f"_⚠️ Enter at market price — adjust levels from your actual entry_"
    )


def format_result_message(symbol: str, direction: str, result: str, pnl_r: float) -> str:
    symbol_clean = symbol.replace(":USDT", "").replace("/USDT", "")
    direction_label = "LONG 🟢" if direction == "long" else "SHORT 🔴"

    if result == "tp1":
        return (
            f"⚡ *TP1 HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *+{pnl_r}R* | Move SL to entry now"
        )
    elif result == "tp2":
        return (
            f"🎯 *TP2 HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *+{pnl_r}R* | Full target reached ✅"
        )
    elif result == "sl":
        return (
            f"❌ *SL HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *{pnl_r}R* | Loss taken"
        )
    else:
        return (
            f"⏱ *EXPIRED* — {direction_label} {symbol_clean}\n"
            f"Signal closed without hitting TP or SL"
        )


async def send_signal(bot_token: str, chat_id: str, sig: Signal, df: pd.DataFrame = None):
    bot = Bot(token=bot_token)

    # Send chart first
    if df is not None:
        try:
            from bot.charting import generate_chart
            chart_buf = generate_chart(df, sig)
            if chart_buf:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=chart_buf,
                )
        except Exception as e:
            logger.error(f"Chart send failed: {e}")

    # Send clean signal text
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=format_signal_message(sig),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Signal message failed: {e}")


async def send_result(bot_token: str, chat_id: str, symbol: str, direction: str, result: str, pnl_r: float):
    bot = Bot(token=bot_token)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=format_result_message(symbol, direction, result, pnl_r),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Result message failed: {e}")
