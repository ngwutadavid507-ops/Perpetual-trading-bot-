"""
Telegram notification sender for Phoenix V2.
Sends signal alerts, result notifications and reversal alerts.
Includes ROI % in all result messages.
"""

import logging
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode

from bot.signal_engine import Signal

logger = logging.getLogger(__name__)


def format_signal_message(sig: Signal) -> str:
    arrow = "🟢 LONG" if sig.direction == "long" else "🔴 SHORT"
    symbol_clean = sig.symbol.replace(":USDT", "").replace("/USDT", "")
    strategy_label = "📈 CONTINUATION" if sig.strategy_type == "continuation" else "🔄 REVERSAL"
    obs_label = " ✅ OBSERVED" if sig.observation_confirmed else " ⚡ IMMEDIATE"

    if sig.direction == "long":
        sl_label = f"`{sig.stop_loss}` (-{sig.sl_pct}%)"
        tp1_label = f"`{sig.take_profit1}` (+{sig.tp1_pct}%)"
        tp2_label = f"`{sig.take_profit2}` (+{sig.tp2_pct}%)"
        tp3_label = f"`{sig.take_profit3}` (+{sig.tp3_pct}%)"
    else:
        sl_label = f"`{sig.stop_loss}` (+{sig.sl_pct}%)"
        tp1_label = f"`{sig.take_profit1}` (-{sig.tp1_pct}%)"
        tp2_label = f"`{sig.take_profit2}` (-{sig.tp2_pct}%)"
        tp3_label = f"`{sig.take_profit3}` (-{sig.tp3_pct}%)"

    # ROI calculations at each TP level
    tp1_roi = round(sig.tp1_pct * sig.leverage, 1)
    tp2_roi = round(sig.tp2_pct * sig.leverage, 1)
    tp3_roi = round(sig.tp3_pct * sig.leverage, 1)

    reasons_block = "\n".join(f"• {r}" for r in sig.reasons)

    # News sentiment indicator
    news_icon = ""
    if sig.news_sentiment == "bullish" and sig.direction == "long":
        news_icon = "📰 News: Bullish ✅\n"
    elif sig.news_sentiment == "bearish" and sig.direction == "short":
        news_icon = "📰 News: Bearish ✅\n"
    elif sig.news_sentiment != "neutral":
        news_icon = "📰 News: Neutral\n"

    return (
        f"*{arrow} — {symbol_clean}*\n"
        f"{strategy_label}{obs_label} | Confidence: *{sig.confidence}%* | Leverage: *{sig.leverage}x*\n"
        f"{news_icon}\n"
        f"Entry:  `{sig.entry}`\n"
        f"SL:     {sl_label}\n\n"
        f"TP1:    {tp1_label} → *+{tp1_roi}% ROI* | close 30%\n"
        f"TP2:    {tp2_label} → *+{tp2_roi}% ROI* | close 30%\n"
        f"TP3:    {tp3_label} → *+{tp3_roi}% ROI* | close 40%\n\n"
        f"{reasons_block}\n\n"
        f"_⚠️ Enter at market price — adjust levels from your actual entry_"
    )


def format_result_message(
    symbol: str,
    direction: str,
    result: str,
    pnl_r: float,
    leverage: int = 1,
    roi_pct: float = 0.0,
) -> str:
    symbol_clean = symbol.replace(":USDT", "").replace("/USDT", "")
    direction_label = "LONG 🟢" if direction == "long" else "SHORT 🔴"

    roi_str = f"+{roi_pct}%" if roi_pct >= 0 else f"{roi_pct}%"
    r_str = f"+{pnl_r}R" if pnl_r >= 0 else f"{pnl_r}R"

    if result == "tp1":
        return (
            f"⚡ *TP1 HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *{r_str}* | ROI: *{roi_str}* ({leverage}x)\n"
            f"Move SL to entry — let rest run to TP2"
        )
    elif result == "tp2":
        return (
            f"🎯 *TP2 HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *{r_str}* | ROI: *{roi_str}* ({leverage}x)\n"
            f"Close another 30% — trail remaining to TP3"
        )
    elif result == "tp3":
        return (
            f"🏆 *TP3 HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *{r_str}* | ROI: *{roi_str}* ({leverage}x)\n"
            f"Full target reached ✅"
        )
    elif result == "sl":
        return (
            f"❌ *SL HIT* — {direction_label} {symbol_clean}\n"
            f"Result: *{r_str}* | ROI: *{roi_str}* ({leverage}x)\n"
            f"Loss taken — wait for next signal"
        )
    else:
        return (
            f"⏱ *EXPIRED* — {direction_label} {symbol_clean}\n"
            f"Signal closed without hitting any target"
        )


def format_reversal_message(
    active_symbol: str,
    active_direction: str,
    new_sig: Signal,
) -> str:
    symbol_clean = active_symbol.replace(":USDT", "").replace("/USDT", "")
    active_label = "LONG 🟢" if active_direction == "long" else "SHORT 🔴"
    new_label = "LONG 🟢" if new_sig.direction == "long" else "SHORT 🔴"

    if active_direction == "long":
        action = "Close at least 50% of your LONG — take partial profit now"
    else:
        action = "Close at least 50% of your SHORT — take partial profit now"

    if new_sig.direction == "long":
        sl_label = f"`{new_sig.stop_loss}` (-{new_sig.sl_pct}%)"
        tp1_label = f"`{new_sig.take_profit1}` (+{new_sig.tp1_pct}%)"
        tp2_label = f"`{new_sig.take_profit2}` (+{new_sig.tp2_pct}%)"
        tp3_label = f"`{new_sig.take_profit3}` (+{new_sig.tp3_pct}%)"
    else:
        sl_label = f"`{new_sig.stop_loss}` (+{new_sig.sl_pct}%)"
        tp1_label = f"`{new_sig.take_profit1}` (-{new_sig.tp1_pct}%)"
        tp2_label = f"`{new_sig.take_profit2}` (-{new_sig.tp2_pct}%)"
        tp3_label = f"`{new_sig.take_profit3}` (-{new_sig.tp3_pct}%)"

    tp1_roi = round(new_sig.tp1_pct * new_sig.leverage, 1)
    tp2_roi = round(new_sig.tp2_pct * new_sig.leverage, 1)
    tp3_roi = round(new_sig.tp3_pct * new_sig.leverage, 1)

    reasons_block = "\n".join(f"• {r}" for r in new_sig.reasons)

    return (
        f"⚠️ *REVERSAL ALERT — {symbol_clean}*\n\n"
        f"Active trade: *{active_label}*\n"
        f"{action}\n\n"
        f"*New signal: {new_label}*\n"
        f"Confidence: *{new_sig.confidence}%* | Leverage: *{new_sig.leverage}x*\n\n"
        f"Entry:  `{new_sig.entry}`\n"
        f"SL:     {sl_label}\n\n"
        f"TP1:    {tp1_label} → *+{tp1_roi}% ROI* | close 30%\n"
        f"TP2:    {tp2_label} → *+{tp2_roi}% ROI* | close 30%\n"
        f"TP3:    {tp3_label} → *+{tp3_roi}% ROI* | close 40%\n\n"
        f"{reasons_block}\n\n"
        f"_⚠️ Only act on this if you confirm the reversal on your chart_"
    )


async def send_signal(
    bot_token: str,
    chat_id: str,
    sig: Signal,
    df: pd.DataFrame = None,
):
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

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=format_signal_message(sig),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Signal message failed: {e}")


async def send_result(
    bot_token: str,
    chat_id: str,
    symbol: str,
    direction: str,
    result: str,
    pnl_r: float,
    leverage: int = 1,
    roi_pct: float = 0.0,
):
    bot = Bot(token=bot_token)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=format_result_message(
                symbol, direction, result,
                pnl_r, leverage, roi_pct
            ),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Result message failed: {e}")


async def send_reversal_alert(
    bot_token: str,
    chat_id: str,
    active_symbol: str,
    active_direction: str,
    new_sig: Signal,
    df: pd.DataFrame = None,
):
    bot = Bot(token=bot_token)

    if df is not None:
        try:
            from bot.charting import generate_chart
            chart_buf = generate_chart(df, new_sig)
            if chart_buf:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=chart_buf,
                )
        except Exception as e:
            logger.error(f"Reversal chart failed: {e}")

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=format_reversal_message(
                active_symbol, active_direction, new_sig
            ),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Reversal message failed: {e}")
