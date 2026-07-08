"""
News sentiment filter using RSS feeds.
Checks CoinDesk, Cointelegraph and Decrypt for recent news
about a token before firing a signal.

Sentiment scoring:
- Strong negative keywords → block signal
- Strong positive keywords → boost confidence
- No relevant news → signal proceeds normally

No API key required — uses free RSS feeds.
"""

import logging
import time
from datetime import datetime, timezone, timedelta

import feedparser

logger = logging.getLogger(__name__)

# Free RSS feeds — no API key needed
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
]

# Keywords that indicate bearish/negative news
BEARISH_KEYWORDS = [
    "hack", "hacked", "exploit", "exploited", "breach", "stolen",
    "sec", "lawsuit", "sued", "ban", "banned", "illegal", "fraud",
    "scam", "rug", "crash", "collapse", "bankrupt", "insolvent",
    "delisted", "delist", "investigation", "arrested", "charges",
    "warning", "alert", "vulnerability", "attack", "lost", "stolen",
    "dump", "dumping", "selloff", "fear", "panic", "concern",
]

# Keywords that indicate bullish/positive news
BULLISH_KEYWORDS = [
    "partnership", "partners", "launch", "launched", "upgrade",
    "etf", "approved", "approval", "adoption", "integration",
    "listing", "listed", "institutional", "investment", "invest",
    "milestone", "record", "growth", "rally", "bullish",
    "mainnet", "testnet", "v2", "protocol", "ecosystem",
    "funding", "raised", "grant", "backing", "support",
    "breakthrough", "innovation", "first", "leading",
]

# Cache: {feed_url: (timestamp, entries)}
_feed_cache: dict[str, tuple[float, list]] = {}
FEED_CACHE_TTL = 300  # 5 minutes


def _fetch_feed(url: str) -> list[dict]:
    """Fetch and cache an RSS feed."""
    now = time.time()
    if url in _feed_cache:
        cached_at, entries = _feed_cache[url]
        if now - cached_at < FEED_CACHE_TTL:
            return entries

    try:
        feed = feedparser.parse(url)
        entries = []
        for entry in feed.entries[:20]:
            entries.append({
                "title": entry.get("title", "").lower(),
                "summary": entry.get("summary", "").lower(),
                "published": entry.get("published_parsed"),
            })
        _feed_cache[url] = (now, entries)
        return entries
    except Exception as e:
        logger.debug(f"[news] feed fetch failed for {url}: {e}")
        return []


def _entry_is_recent(entry: dict, hours: int) -> bool:
    """Check if a feed entry was published within the last N hours."""
    published = entry.get("published")
    if not published:
        return True  # Assume recent if no date

    try:
        pub_time = datetime(*published[:6], tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return pub_time >= cutoff
    except Exception:
        return True


def _mentions_token(entry: dict, token: str) -> bool:
    """Check if an entry mentions the token."""
    token_lower = token.lower()
    return (
        token_lower in entry["title"] or
        token_lower in entry["summary"]
    )


def get_news_sentiment(
    token: str,
    lookback_hours: int = 2,
) -> tuple[str, list[str]]:
    """
    Checks recent news for a token and returns sentiment.

    Returns:
        (sentiment, headlines) where sentiment is:
        'bullish' — positive news found
        'bearish' — negative news found
        'neutral' — no significant news found

    headlines: list of relevant headline strings
    """
    bearish_count = 0
    bullish_count = 0
    headlines = []

    for feed_url in RSS_FEEDS:
        entries = _fetch_feed(feed_url)
        for entry in entries:
            if not _entry_is_recent(entry, lookback_hours):
                continue
            if not _mentions_token(entry, token):
                continue

            title = entry["title"]
            headlines.append(title)

            # Score sentiment
            text = title + " " + entry["summary"]
            for kw in BEARISH_KEYWORDS:
                if kw in text:
                    bearish_count += 1
                    break

            for kw in BULLISH_KEYWORDS:
                if kw in text:
                    bullish_count += 1
                    break

    if not headlines:
        return "neutral", []

    if bearish_count > bullish_count:
        return "bearish", headlines
    elif bullish_count > bearish_count:
        return "bullish", headlines
    else:
        return "neutral", headlines


def should_block_signal(
    token: str,
    direction: str,
    lookback_hours: int = 2,
) -> tuple[bool, str]:
    """
    Determines if a signal should be blocked based on news sentiment.

    Returns:
        (should_block, reason)
        should_block: True if signal should be blocked
        reason: explanation string
    """
    try:
        sentiment, headlines = get_news_sentiment(token, lookback_hours)

        if sentiment == "neutral":
            return False, ""

        if sentiment == "bearish" and direction == "long":
            headline_preview = headlines[0][:80] if headlines else ""
            return True, f"Bearish news detected: {headline_preview}"

        if sentiment == "bullish" and direction == "short":
            headline_preview = headlines[0][:80] if headlines else ""
            return True, f"Bullish news detected: {headline_preview}"

        return False, ""

    except Exception as e:
        logger.error(f"[news] sentiment check failed for {token}: {e}")
        return False, ""


def get_confidence_boost(
    token: str,
    direction: str,
    lookback_hours: int = 2,
) -> float:
    """
    Returns a confidence boost (0-10) if news confirms signal direction.
    """
    try:
        sentiment, headlines = get_news_sentiment(token, lookback_hours)

        if sentiment == "bullish" and direction == "long":
            return 5.0
        if sentiment == "bearish" and direction == "short":
            return 5.0

        return 0.0

    except Exception:
        return 0.0
