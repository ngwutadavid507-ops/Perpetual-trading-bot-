import logging
from telegram import Bot
from telegram.constants import ParseMode

from bot.signal_engine import Signal

logger = logging.getLogger(__name__)


def format_signal_message(sig: Signal) -> str:
    arrow = "🟢 LONG" if sig.direction == "long" else "🔴 SHORT"
    reasons_block = "\n".join(f"• {r}" for r in sig.reasons)
    return (
        f"*{arrow} — {sig.symbol}* ({sig.exchange})\n\n"
        f"Confidence: *{sig.confidence}%*\n"
        f"Entry: `{sig.entry}`\n"
        f"Stop Loss: `{sig.stop_loss}`\n"
        f"Take Profit: `{sig.take_profit}`\n"
        f"Risk:Reward: *1:{sig.risk_reward}*\n\n"
        f"Reasons:\n{reasons_block}\n\n"
        f"_Not financial advice. Confirm on your own chart before entering._"
    )


async def send_signal(bot_token: str, chat_id: str, sig: Signal):
    bot = Bot(token=bot_token)
    text = format_signal_message(sig)
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
