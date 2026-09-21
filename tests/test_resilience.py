"""
Tests for resilience patterns.
"""
import pytest
import time
from cldb.resilience import (
    CircuitBreaker, CircuitState, CircuitOpenError, RateLimiter,
    retry_with_backoff, circuit_breaker, CircuitBreakerRegistry
)


def test_circuit_breaker_closed_state():
    breaker = CircuitBreaker("test", failure_threshold=3)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() == True


def test_circuit_breaker_opens_after_failures():
    breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=0.1)
    
    for _ in range(3):
        breaker.record_failure()
    
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() == False


def test_circuit_breaker_half_open_after_timeout():
    breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=0.1)
    
    for _ in range(3):
        breaker.record_failure()
    
    assert breaker.state == CircuitState.OPEN
    
    time.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.allow_request() == True


def test_circuit_breaker_success_resets():
    breaker = CircuitBreaker("test", failure_threshold=3, half_open_max_calls=2)
    
    for _ in range(3):
        breaker.record_failure()
    
    breaker.record_success()
    breaker.record_success()
    
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_reset():
    breaker = CircuitBreaker("test", failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    
    breaker.reset()
    
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_rate_limiter_initial_tokens():
    limiter = RateLimiter(rate=10, capacity=5)
    assert limiter.available_tokens == 5


def test_rate_limiter_acquire():
    limiter = RateLimiter(rate=10, capacity=5)
    
    assert limiter.acquire(tokens=2) == True
    assert limiter.available_tokens < 5


def test_rate_limiter_no_blocking():
    limiter = RateLimiter(rate=0.1, capacity=1)
    
    limiter.acquire(tokens=1)
    result = limiter.acquire(tokens=1, blocking=False)
    
    assert result == False


def test_retry_with_backoff_success():
    call_count = 0
    
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def success_func():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = success_func()
    assert result == "success"
    assert call_count == 1


def test_retry_with_backoff_failure_then_success():
    call_count = 0
    
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Retry")
        return "success"
    
    result = flaky_func()
    assert result == "success"
    assert call_count == 3


def test_retry_with_backoff_all_fail():
    call_count = 0
    
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fail")
    
    with pytest.raises(ValueError):
        always_fail()
    
    assert call_count == 3


def test_retry_with_fallback():
    call_count = 0
    
    @retry_with_backoff(max_attempts=2, base_delay=0.01, fallback=lambda: "fallback")
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ValueError("Fail")
    
    result = always_fail()
    assert result == "fallback"


def test_circuit_breaker_decorator():
    @circuit_breaker("test", failure_threshold=2, fallback=lambda: "fallback")
    def failing_func():
        raise ValueError("Fail")
    
    result = failing_func()
    assert result == "fallback"


def test_circuit_breaker_registry():
    registry = CircuitBreakerRegistry.get_instance()
    
    breaker1 = registry.get("test1")
    breaker2 = registry.get("test1")
    
    assert breaker1 is breaker2  # Same instance


def test_circuit_open_error():
    with pytest.raises(CircuitOpenError):
        raise CircuitOpenError("Test circuit open")