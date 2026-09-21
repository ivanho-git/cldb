"""Quick test of CLDB core modules"""
import sys
sys.path.insert(0, '.')

# Test modules directly without __init__.py
import importlib.util

def test_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f'{name}: OK')
    return module

# Test each module
config = test_module('config', 'cldb/config.py')
resilience = test_module('resilience', 'cldb/resilience.py')
logger = test_module('logger', 'cldb/logger.py')
metrics = test_module('metrics', 'cldb/metrics.py')

# Functional tests
breaker = resilience.CircuitBreaker('test', failure_threshold=3)
print(f'\nCircuit breaker state: {breaker.state.value}')

counter = metrics.Counter('test_counter', ['action'])
counter.inc(value=5, action='create')
print(f'Counter value: {counter.get(action="create")}')

limiter = resilience.RateLimiter(rate=10, capacity=5)
print(f'Rate limiter tokens: {limiter.available_tokens:.1f}')

histogram = metrics.Histogram('test_hist')
histogram.observe(0.1)
histogram.observe(0.5)
print('Histogram observation: OK')

# Test retry decorator
call_count = [0]

@resilience.retry_with_backoff(max_attempts=3, base_delay=0.01)
def test_func():
    call_count[0] += 1
    if call_count[0] < 2:
        raise ValueError("test")
    return "success"

result = test_func()
print(f'Retry decorator result: {result}, calls: {call_count[0]}')

# Test config
c = config.Config()
print(f'Config environment: {c.environment}')
print(f'Config ML LR: {c.ml.learning_rate}')

print('\nAll core module tests passed!')