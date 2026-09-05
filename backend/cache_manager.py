import os
import time
import json
import hashlib
import urllib.request
import urllib.error
from collections import OrderedDict
from threading import Lock
from typing import Optional, Dict, Any, List

class InMemoryLRUCache:
    """Thread-safe in-memory LRU Cache with TTL fallback (Zero dependency, $0 cost)."""
    def __init__(self, capacity: int = 1000, default_ttl: int = 86400):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            item = self.cache[key]
            if time.time() > item["expires_at"]:
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return item["val"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = {
                "val": value,
                "expires_at": time.time() + ttl
            }
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def size(self) -> int:
        with self.lock:
            return len(self.cache)

class CacheManager:
    """
    Universal High-Performance Cache Manager.
    Supports Upstash Serverless Redis REST API / Redis URL with automatic In-Memory LRU fallback.
    """
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        self.memory_cache = InMemoryLRUCache(capacity=2000, default_ttl=86400)
        self.rate_history: Dict[str, List[float]] = {}
        self.rate_lock = Lock()

    def get_backend_name(self) -> str:
        if self.upstash_url and self.upstash_token:
            return "upstash_rest"
        if self.redis_url:
            return "redis"
        return "in_memory_lru"

    @staticmethod
    def compute_query_hash(query_text: str, image_bytes_list: List[bytes]) -> str:
        hasher = hashlib.sha256()
        hasher.update(query_text.strip().lower().encode("utf-8"))
        for img in image_bytes_list:
            # Hash up to first 2MB or full image to keep hashing sub-millisecond
            sample = img if len(img) <= 2 * 1024 * 1024 else img[: 1024 * 1024] + img[-1024 * 1024 :]
            hasher.update(hashlib.sha256(sample).digest())
        return f"satquery:{hasher.hexdigest()}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        # 1. Try Upstash REST API if configured
        if self.upstash_url and self.upstash_token:
            try:
                url = f"{self.upstash_url.rstrip('/')}/get/{key}"
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {self.upstash_token}"}
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    result = data.get("result")
                    if result:
                        return json.loads(result) if isinstance(result, str) else result
            except Exception as e:
                # Silently fallback to memory cache
                pass

        # 2. In-memory LRU fallback
        return self.memory_cache.get(key)

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 86400) -> bool:
        # Always store in memory cache
        self.memory_cache.set(key, value, ttl=ttl_seconds)

        # Also store in Upstash if configured
        if self.upstash_url and self.upstash_token:
            try:
                url = f"{self.upstash_url.rstrip('/')}/set/{key}"
                payload = json.dumps(value)
                req = urllib.request.Request(
                    f"{url}?ex={ttl_seconds}",
                    data=payload.encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.upstash_token}",
                        "Content-Type": "application/json"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    return resp.status == 200
            except Exception:
                pass
        return True

    def check_rate_limit(self, client_ip: str, limit: int = 60, window_seconds: int = 60) -> bool:
        """
        Sliding-window rate limiter. Returns True if request is allowed, False if exceeded.
        """
        now = time.time()
        window_start = now - window_seconds
        with self.rate_lock:
            timestamps = [t for t in self.rate_history.get(client_ip, []) if t > window_start]
            if len(timestamps) >= limit:
                return False
            timestamps.append(now)
            self.rate_history[client_ip] = timestamps
            return True

# Global singleton
cache_manager = CacheManager()
