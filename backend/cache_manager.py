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
    """Thread-safe in-memory LRU Cache with TTL fallback (Zero external dependency, $0 cost)."""
    def __init__(self, capacity: int = 2000, default_ttl: int = 86400):
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
    Universal High-Performance Redis & Memory Cache Manager.
    - Native Redis (redis-py client via REDIS_URL or redis://localhost:6379)
    - Upstash Serverless Redis REST API fallback
    - Thread-safe in-memory LRU fallback
    - Distributed Redis rate-limiting (atomic INCR + EXPIRE)
    """
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        self.memory_cache = InMemoryLRUCache(capacity=2000, default_ttl=86400)
        self.rate_history: Dict[str, List[float]] = {}
        self.rate_lock = Lock()
        self._redis_client = None
        self._redis_connected = False
        self._init_redis()

    def _init_redis(self):
        """Initializes native Redis connection if available."""
        try:
            import redis
            if self.redis_url:
                client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.5
                )
                # Quick health ping
                client.ping()
                self._redis_client = client
                self._redis_connected = True
                print(f"[Redis Cache] Successfully connected to native Redis at {self.redis_url}")
        except Exception as e:
            self._redis_client = None
            self._redis_connected = False
            # Silently fallback to Upstash or In-Memory
            # (No crash if local Redis daemon is not running)

    def is_redis_connected(self) -> bool:
        if not self._redis_connected or not self._redis_client:
            return False
        try:
            self._redis_client.ping()
            return True
        except Exception:
            self._redis_connected = False
            return False

    def get_backend_name(self) -> str:
        if self.is_redis_connected():
            return "native_redis"
        if self.upstash_url and self.upstash_token:
            return "upstash_redis_rest"
        return "in_memory_lru"

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry on caching layers."""
        active_backend = self.get_backend_name()
        return {
            "active_backend": active_backend,
            "native_redis": {
                "configured_url": self.redis_url if self.redis_url else None,
                "connected": self.is_redis_connected(),
            },
            "upstash_rest": {
                "configured": bool(self.upstash_url and self.upstash_token),
            },
            "in_memory_lru": {
                "size": self.memory_cache.size(),
                "capacity": self.memory_cache.capacity
            }
        }

    @staticmethod
    def compute_query_hash(query_text: str, image_bytes_list: List[bytes]) -> str:
        hasher = hashlib.sha256()
        hasher.update(query_text.strip().lower().encode("utf-8"))
        for img in image_bytes_list:
            sample = img if len(img) <= 2 * 1024 * 1024 else img[: 1024 * 1024] + img[-1024 * 1024 :]
            hasher.update(hashlib.sha256(sample).digest())
        return f"satquery:{hasher.hexdigest()}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        # 1. Try Native Redis
        if self.is_redis_connected():
            try:
                data = self._redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                self._redis_connected = False

        # 2. Try Upstash REST API if configured
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
            except Exception:
                pass

        # 3. In-memory LRU fallback
        return self.memory_cache.get(key)

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 86400) -> bool:
        # Always store in memory cache
        self.memory_cache.set(key, value, ttl=ttl_seconds)

        # Also store in Native Redis if connected
        if self.is_redis_connected():
            try:
                payload = json.dumps(value)
                self._redis_client.setex(key, ttl_seconds, payload)
            except Exception:
                self._redis_connected = False

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
        Rate limiter supporting Redis atomic INCR + EXPIRE across distributed instances,
        with automatic fallback to in-memory sliding window.
        """
        # Distributed Redis rate limiting
        if self.is_redis_connected():
            try:
                current_window = int(time.time() // window_seconds)
                redis_key = f"ratelimit:{client_ip}:{current_window}"
                count = self._redis_client.incr(redis_key)
                if count == 1:
                    self._redis_client.expire(redis_key, window_seconds + 5)
                return count <= limit
            except Exception:
                self._redis_connected = False

        # In-memory sliding window fallback
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
