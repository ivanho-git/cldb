"""
Resilience patterns for CLDB: retry logic, circuit breakers, and fallback handling.
"""
import time
import threading
from typing import Callable, TypeVar, Optional, Any, Dict
from functools import wraps
from datetime import datetime, timezone
from enum import Enum
import random


T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 half_open_max_calls: int = 3, expected_exception: type = Exception):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.expected_exception = expected_exception
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
            return self._state
    
    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count
    
    def record_success(self):
        with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
    
    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
    
    def allow_request(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        return True
                return False
            elif self._state == CircuitState.HALF_OPEN:
                return self._half_open_calls < self.half_open_max_calls
            return False
    
    def reset(self):
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None


class CircuitOpenError(Exception):
    pass


def circuit_breaker(name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                    fallback: Optional[Callable[..., T]] = None):
    breaker = CircuitBreaker(name, failure_threshold, recovery_timeout)
    
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            if not breaker.allow_request():
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit '{name}' is open")
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                if fallback:
                    return fallback(*args, **kwargs)
                raise
        wrapper.circuit_breaker = breaker
        return wrapper
    return decorator


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0,
                       exponential_base: float = 2.0, jitter: bool = True, retry_on: tuple = (Exception,),
                       fallback: Optional[Callable[..., T]] = None):
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    time.sleep(delay)
            if fallback:
                return fallback(*args, **kwargs)
            raise last_exception
        return wrapper
    return decorator


class RateLimiter:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = None) -> bool:
        start_time = time.time()
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                if not blocking:
                    return False
                wait_time = (tokens - self._tokens) / self.rate
                if timeout:
                    elapsed = time.time() - start_time
                    if elapsed + wait_time > timeout:
                        return False
            time.sleep(min(wait_time, 0.1))
            if timeout:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self._last_update
        new_tokens = elapsed * self.rate
        self._tokens = min(self._tokens + new_tokens, self.capacity)
        self._last_update = now
    
    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class CircuitBreakerRegistry:
    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'CircuitBreakerRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get(self, name: str, **kwargs) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **kwargs)
            return self._breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict]:
        with self._lock:
            return {name: {"state": breaker.state.value, "failure_count": breaker.failure_count}
                    for name, breaker in self._breakers.items()}


def get_vector_store_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get_instance().get("vector_store", failure_threshold=3, recovery_timeout=30.0)


def get_database_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get_instance().get("database", failure_threshold=5, recovery_timeout=60.0)


def get_embedding_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get_instance().get("embedding", failure_threshold=3, recovery_timeout=30.0)


def vector_store_fallback(*args, **kwargs) -> list:
    return []


def database_fallback(*args, **kwargs):
    return None


def embedding_fallback(*args, **kwargs):
    import numpy as np
    return np.zeros(384, dtype=np.float32)