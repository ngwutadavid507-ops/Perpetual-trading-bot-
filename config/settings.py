import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    EXCHANGES = [e.strip() for e in os.getenv("EXCHANGES", "bybit,okx,bingx").split(",") if e.strip()]

    TIMEFRAME = os.getenv("TIMEFRAME", "15m")

    MIN_24H_VOLUME_USDT = float(os.getenv("MIN_24H_VOLUME_USDT", "5000000"))

    MIN_CONFIDENCE_DEFAULT = float(os.getenv("MIN_CONFIDENCE_DEFAULT", "65"))
    MIN_CONFIDENCE_BTC = float(os.getenv("MIN_CONFIDENCE_BTC", "85"))

    SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))

    MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", "1.5"))

    # Symbols treated with the stricter BTC-style threshold
    STRICT_SYMBOLS = {"BTC/USDT:USDT", "BTC/USDT"}

    @classmethod
    def validate(cls):
        missing = []
        if not cls.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            raise RuntimeError(f"Missing required .env values: {', '.join(missing)}")
