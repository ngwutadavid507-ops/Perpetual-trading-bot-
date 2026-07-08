"""
Fetches the top 200 cryptocurrencies by market cap from CoinGecko.
Falls back to a hardcoded list if CoinGecko is unavailable.
Cache persists for 4 hours to avoid rate limits.
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

_cache: tuple[float, set[str]] | None = None
CACHE_TTL_SECONDS = 4 * 60 * 60

FALLBACK_TOP_SYMBOLS = {
    # Top 100
    "BTC", "ETH", "USDT", "BNB", "SOL", "XRP", "USDC", "DOGE", "ADA",
    "TRX", "AVAX", "SHIB", "TON", "LINK", "DOT", "BCH", "NEAR", "LTC",
    "UNI", "ICP", "DAI", "APT", "ATOM", "POL", "ETC", "XLM", "OP",
    "HBAR", "FIL", "ARB", "VET", "MKR", "IMX", "AAVE", "ALGO", "STX",
    "TAO", "SUI", "RENDER", "INJ", "GRT", "FTM", "SAND", "MANA", "AXS",
    "THETA", "XTZ", "EGLD", "FLOW", "KAVA", "NEO", "CHZ", "CRV", "ZEC",
    "COMP", "YFI", "SNX", "1INCH", "RUNE", "LDO", "CAKE", "DYDX", "ENJ",
    "BAT", "ZRX", "OCEAN", "BAND", "KNC", "NMR", "SUSHI", "UMA",
    "CELO", "SKL", "STORJ", "ANKR", "CTSI", "PERP", "SLP",
    "JASMY", "GALA", "ENS", "APE", "GMT", "HNT", "ROSE", "ONE",
    "ZIL", "IOTA", "HOT", "BTT", "WIN", "FLOKI", "PEPE", "WIF",
    # Top 200
    "BONK", "JTO", "PYTH", "JUP", "STRK", "TIA", "HYPE", "WLD",
    "PENDLE", "BLUR", "CFX", "MANTA", "ALT", "PIXEL", "PORTAL",
    "ETHENA", "ENA", "OMNI", "REZ", "SAGA", "TAIKO", "ZK", "LISTA",
    "IO", "ZRO", "BANANA", "DOGS", "HMSTR", "CATI", "MAJOR", "NEIRO",
    "GOAT", "MOODENG", "PNUT", "ACT", "GRASS", "BOME", "SLERF",
    "POPCAT", "MEW", "BRETT", "TURBO", "MOG", "PONKE", "GIGA",
    "FWOG", "MICHI", "KEYCAT", "SUNDOG", "LAUNCHCOIN", "VINE",
    "FARTCOIN", "ZEREBRO", "VIRTUAL", "AIXBT", "MORPHO", "USUAL",
    "RESOLV", "DEEP", "WAL", "ANIME", "FORM", "IP", "PARTI",
    "TST", "KAITO", "SHELL", "MOVE", "COOKIE", "SKYAI",
    "OM", "FET", "AGIX", "AKT", "GNO", "RPL", "LPT",
    "API3", "TRB", "DIA", "HOOK", "LOKA", "CHESS", "UNFI",
    "QUICK", "NULS", "MTL", "POLS", "POND", "HARD", "STMX",
    "WAN", "ARPA", "CTXC", "FOR", "AKRO", "BEL", "SFP",
    "MDT", "PROM", "FRONT", "MBL", "FIRO", "SYS", "VIB",
    "TROY", "PERL", "IDEX", "DOCK", "PHB", "BTS", "KEY",
    "XVS", "ALPHA", "AUTO", "BAKE", "BELT", "TWT", "XNO",
    "MINA", "SCRT", "MOVR", "ACA", "KSM", "ASTR", "PHA",
    "GLMR", "KAR", "WIF", "POPCAT", "ETHFI", "PAXG",
    "OKB", "KCS", "HT", "CRO", "FTT", "BNX", "LEVER",
    "SPX", "PI", "KAS", "NOT", "DOGS", "HMSTR",
    "ZRO", "IO", "LISTA", "TAIKO", "ZK", "SAGA",
}


def get_top_symbols(limit: int = 200) -> set[str]:
    global _cache

    if _cache is not None:
        cached_at, symbols = _cache
        if time.time() - cached_at < CACHE_TTL_SECONDS:
            logger.info(f"[toplist] using cached list ({len(symbols)} symbols)")
            return symbols

    symbols = _fetch_from_coingecko(limit)

    if not symbols:
        if _cache is not None:
            logger.warning("[toplist] fetch failed — using stale cache")
            return _cache[1]
        logger.warning("[toplist] fetch failed — using hardcoded fallback")
        return FALLBACK_TOP_SYMBOLS

    _cache = (time.time(), symbols)
    logger.info(f"[toplist] refreshed — {len(symbols)} top symbols loaded")
    return symbols


def _fetch_from_coingecko(limit: int) -> set[str]:
    symbols = set()
    per_page = min(limit, 250)
    page = 1

    while len(symbols) < limit:
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": False,
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for coin in data:
                symbol = coin.get("symbol", "").upper()
                if symbol:
                    symbols.add(symbol)

            if len(data) < per_page:
                break

            page += 1

        except requests.exceptions.RequestException as e:
            logger.error(f"[toplist] CoinGecko request failed: {e}")
            break
        except Exception as e:
            logger.error(f"[toplist] unexpected error: {e}")
            break

    return symbols


def is_top_symbol(base_currency: str, top_symbols: set[str]) -> bool:
    return base_currency.upper() in top_symbols
