"""
Upstash Redis client using REST API.
Persistent storage that survives Railway restarts.
All state — signal cooldown, daily limits, tracker — stored here.
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")


def _headers() -> dict:
    return {"Authorization": f"Bearer {REDIS_TOKEN}"}


def redis_set(key: str, value: str, ex: int = None) -> bool:
    """Set a key with optional expiry in seconds."""
    try:
        if ex:
            url = f"{REDIS_URL}/set/{key}/{value}/ex/{ex}"
        else:
            url = f"{REDIS_URL}/set/{key}/{value}"
        r = requests.get(url, headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"[redis] set failed for {key}: {e}")
        return False


def redis_get(key: str) -> str | None:
    """Get a key value or None if not exists."""
    try:
        url = f"{REDIS_URL}/get/{key}"
        r = requests.get(url, headers=_headers(), timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("result")
        return None
    except Exception as e:
        logger.error(f"[redis] get failed for {key}: {e}")
        return None


def redis_delete(key: str) -> bool:
    """Delete a key."""
    try:
        url = f"{REDIS_URL}/del/{key}"
        r = requests.get(url, headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"[redis] delete failed for {key}: {e}")
        return False


def redis_incr(key: str) -> int | None:
    """Increment a counter and return new value."""
    try:
        url = f"{REDIS_URL}/incr/{key}"
        r = requests.get(url, headers=_headers(), timeout=5)
        if r.status_code == 200:
            return r.json().get("result")
        return None
    except Exception as e:
        logger.error(f"[redis] incr failed for {key}: {e}")
        return None


def redis_expire(key: str, seconds: int) -> bool:
    """Set expiry on existing key."""
    try:
        url = f"{REDIS_URL}/expire/{key}/{seconds}"
        r = requests.get(url, headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"[redis] expire failed for {key}: {e}")
        return False


def redis_hset(key: str, field: str, value: str) -> bool:
    """Set a hash field."""
    try:
        url = f"{REDIS_URL}/hset/{key}/{field}/{value}"
        r = requests.get(url, headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"[redis] hset failed for {key}/{field}: {e}")
        return False


def redis_hget(key: str, field: str) -> str | None:
    """Get a hash field."""
    try:
        url = f"{REDIS_URL}/hget/{key}/{field}"
        r = requests.get(url, headers=_headers(), timeout=5)
        if r.status_code == 200:
            return r.json().get("result")
        return None
    except Exception as e:
        logger.error(f"[redis] hget failed for {key}/{field}: {e}")
        return None


def redis_hgetall(key: str) -> dict:
    """Get all hash fields and values."""
    try:
        url = f"{REDIS_URL}/hgetall/{key}"
        r = requests.get(url, headers=_headers(), timeout=5)
        if r.status_code == 200:
            result = r.json().get("result", [])
            # Upstash returns flat list [field, value, field, value...]
            return dict(zip(result[::2], result[1::2])) if result else {}
        return {}
    except Exception as e:
        logger.error(f"[redis] hgetall failed for {key}: {e}")
        return {}


def redis_hdel(key: str, field: str) -> bool:
    """Delete a hash field."""
    try:
        url = f"{REDIS_URL}/hdel/{key}/{field}"
        r = requests.get(url, headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"[redis] hdel failed for {key}/{field}: {e}")
        return False
