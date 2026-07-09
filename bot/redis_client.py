"""
Upstash Redis client using REST API with POST requests.
Handles complex JSON values safely.
All bot state lives here — survives Railway restarts.
"""

import logging
import os
import requests
import json

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {REDIS_TOKEN}",
        "Content-Type": "application/json",
    }


def _post(command: list) -> dict:
    try:
        r = requests.post(
            REDIS_URL,
            headers=_headers(),
            json=command,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        logger.error(f"[redis] HTTP {r.status_code}: {r.text}")
        return {}
    except Exception as e:
        logger.error(f"[redis] request failed: {e}")
        return {}


def redis_set(key: str, value: str, ex: int = None) -> bool:
    if ex:
        result = _post(["SET", key, value, "EX", ex])
    else:
        result = _post(["SET", key, value])
    return result.get("result") == "OK"


def redis_get(key: str) -> str | None:
    result = _post(["GET", key])
    return result.get("result")


def redis_delete(key: str) -> bool:
    result = _post(["DEL", key])
    return bool(result.get("result"))


def redis_incr(key: str) -> int | None:
    result = _post(["INCR", key])
    return result.get("result")


def redis_expire(key: str, seconds: int) -> bool:
    result = _post(["EXPIRE", key, seconds])
    return bool(result.get("result"))


def redis_hset(key: str, field: str, value: str) -> bool:
    result = _post(["HSET", key, field, value])
    return result.get("result") is not None


def redis_hget(key: str, field: str) -> str | None:
    result = _post(["HGET", key, field])
    return result.get("result")


def redis_hgetall(key: str) -> dict:
    result = _post(["HGETALL", key])
    raw = result.get("result", [])
    if not raw or not isinstance(raw, list):
        return {}
    pairs = {}
    it = iter(raw)
    for field in it:
        try:
            value = next(it)
            if isinstance(field, str) and not field.startswith("{"):
                pairs[field] = value
        except StopIteration:
            break
    return pairs


def redis_hdel(key: str, field: str) -> bool:
    result = _post(["HDEL", key, field])
    return bool(result.get("result"))


def redis_keys(pattern: str) -> list[str]:
    result = _post(["KEYS", pattern])
    return result.get("result", [])


def redis_flushdb() -> bool:
    result = _post(["FLUSHDB"])
    return result.get("result") == "OK"
