import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # Exchanges
    EXCHANGES = [
        e.strip() for e in
        os.getenv("EXCHANGES", "okx,bingx").split(",")
        if e.strip()
    ]

    # Timeframes
    TIMEFRAME_TREND = "1h"       # Higher timeframe for trend direction
    TIMEFRAME_ENTRY = "5m"       # Lower timeframe for entry signals

    # Volume filter
    MIN_24H_VOLUME_USDT = float(os.getenv("MIN_24H_VOLUME_USDT", "3000000"))

    # Confidence thresholds
    MIN_CONFIDENCE_CONTINUATION = float(os.getenv("MIN_CONFIDENCE_CONTINUATION", "65"))
    MIN_CONFIDENCE_REVERSAL = float(os.getenv("MIN_CONFIDENCE_REVERSAL", "80"))

    # Signal limits
    MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "15"))
    MAX_SIGNALS_PER_SCAN = int(os.getenv("MAX_SIGNALS_PER_SCAN", "2"))

    # Timing
    SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
    SIGNAL_COOLDOWN_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "60"))

    # Risk
    MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", "2.0"))

    # Live price validation
    # Max % price can move from signal detection to delivery before signal is discarded
    MAX_ENTRY_SLIPPAGE_PCT = float(os.getenv("MAX_ENTRY_SLIPPAGE_PCT", "0.3"))

    # Cross exchange confirmation
    # Single exchange min confidence to fire without second exchange confirmation
    SINGLE_EXCHANGE_MIN_CONFIDENCE = float(os.getenv("SINGLE_EXCHANGE_MIN_CONFIDENCE", "75"))

    # Redis
    UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
    UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

    # News sentiment
    NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "2"))
    NEWS_BLOCK_ON_NEGATIVE = os.getenv("NEWS_BLOCK_ON_NEGATIVE", "true").lower() == "true"

    # Junk tokens — always blocked regardless of top list
    JUNK_EXACT = {
        "PIEVERSE", "WLFI", "EURI", "EURC", "FDUSD",
        "TUSD", "BUSD", "USDP", "GUSD", "HUSD",
        "NIGHT", "CC", "MON", "ASTER", "SKY", "ACE",
        "LEVER", "COMBO", "VIDT", "HIGH", "LAZIO",
        "PORTO", "ALPINE", "CITY", "SANTOS", "PSG",
        "ATM", "OG", "TBT", "INTER", "JUV", "BAR",
        "REAL", "REALT", "TRUMP", "MELANIA", "MAGA",
        "ZZ", "2Z", "ZBCN",
    }

    JUNK_PREFIXES = [
        "NC", "EUR", "GBP", "SGD", "JPY", "AUD",
        "USD2", "2USD", "BVOL", "DVOL",
    ]

    @classmethod
    def validate(cls):
        missing = []
        if not cls.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if not cls.UPSTASH_REDIS_REST_URL:
            missing.append("UPSTASH_REDIS_REST_URL")
        if not cls.UPSTASH_REDIS_REST_TOKEN:
            missing.append("UPSTASH_REDIS_REST_TOKEN")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
        )
