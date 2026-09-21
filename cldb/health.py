"""
Health endpoints and server for CLDB monitoring.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import threading
import time

from sqlalchemy import text

from cldb.logger import get_logger
from cldb.resilience import CircuitBreakerRegistry


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: Optional[str] = None


@dataclass
class SystemHealth:
    status: HealthStatus
    timestamp: str
    uptime_seconds: float
    components: List[ComponentHealth]
    version: str = "1.0.0"
    environment: str = "development"


class HealthChecker:
    def __init__(self, repo=None, vector_store=None, cl_engine=None, embedder=None):
        self.logger = get_logger("health")
        self._repo = repo
        self._vector_store = vector_store
        self._cl_engine = cl_engine
        self._embedder = embedder
        self._start_time = time.time()
        self._circuit_breakers = CircuitBreakerRegistry.get_instance()
    
    def check_database(self) -> ComponentHealth:
        start = time.time()
        try:
            if self._repo:
                with self._repo.get_session() as session:
                    session.execute(text("SELECT 1"))
                latency = (time.time() - start) * 1000
                return ComponentHealth(name="database", status=HealthStatus.HEALTHY, latency_ms=latency,
                                       last_check=datetime.now(timezone.utc).isoformat())
            else:
                return ComponentHealth(name="database", status=HealthStatus.DEGRADED, message="No database configured",
                                       last_check=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            return ComponentHealth(name="database", status=HealthStatus.UNHEALTHY, message=str(e),
                                   last_check=datetime.now(timezone.utc).isoformat())
    
    def check_vector_store(self) -> ComponentHealth:
        start = time.time()
        try:
            if self._vector_store:
                health = self._vector_store.health_check()
                latency = (time.time() - start) * 1000
                if health.get("status") == "healthy":
                    return ComponentHealth(name="vector_store", status=HealthStatus.HEALTHY, latency_ms=latency,
                                           last_check=datetime.now(timezone.utc).isoformat())
                elif health.get("status") == "using_fallback":
                    return ComponentHealth(name="vector_store", status=HealthStatus.DEGRADED, latency_ms=latency,
                                           message="Using in-memory fallback", last_check=datetime.now(timezone.utc).isoformat())
                else:
                    return ComponentHealth(name="vector_store", status=HealthStatus.UNHEALTHY, message=health.get("error", "Unknown error"),
                                           last_check=datetime.now(timezone.utc).isoformat())
            else:
                return ComponentHealth(name="vector_store", status=HealthStatus.DEGRADED, message="No vector store configured",
                                       last_check=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            return ComponentHealth(name="vector_store", status=HealthStatus.UNHEALTHY, message=str(e),
                                   last_check=datetime.now(timezone.utc).isoformat())
    
    def check_model(self) -> ComponentHealth:
        try:
            if self._cl_engine:
                return ComponentHealth(name="ml_model", status=HealthStatus.HEALTHY, message="Model loaded and ready",
                                       last_check=datetime.now(timezone.utc).isoformat())
            else:
                return ComponentHealth(name="ml_model", status=HealthStatus.DEGRADED, message="No model loaded",
                                       last_check=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            return ComponentHealth(name="ml_model", status=HealthStatus.UNHEALTHY, message=str(e),
                                   last_check=datetime.now(timezone.utc).isoformat())
    
    def check_embeddings(self) -> ComponentHealth:
        start = time.time()
        try:
            if self._embedder:
                test_emb = self._embedder.embed("SELECT 1")
                latency = (time.time() - start) * 1000
                if test_emb is not None and len(test_emb) > 0:
                    return ComponentHealth(name="embeddings", status=HealthStatus.HEALTHY, latency_ms=latency,
                                           last_check=datetime.now(timezone.utc).isoformat())
                return ComponentHealth(name="embeddings", status=HealthStatus.UNHEALTHY, message="Embedding returned empty result",
                                       last_check=datetime.now(timezone.utc).isoformat())
            else:
                return ComponentHealth(name="embeddings", status=HealthStatus.DEGRADED, message="No embedder configured",
                                       last_check=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            return ComponentHealth(name="embeddings", status=HealthStatus.UNHEALTHY, message=str(e),
                                   last_check=datetime.now(timezone.utc).isoformat())
    
    def check_circuit_breakers(self) -> List[ComponentHealth]:
        statuses = self._circuit_breakers.get_all_status()
        components = []
        for name, status in statuses.items():
            cb_status = status["state"]
            if cb_status == "closed":
                component_status = HealthStatus.HEALTHY
            elif cb_status == "half_open":
                component_status = HealthStatus.DEGRADED
            else:
                component_status = HealthStatus.UNHEALTHY
            components.append(ComponentHealth(name=f"circuit_breaker_{name}", status=component_status,
                                              message=f"State: {cb_status}, Failures: {status['failure_count']}",
                                              last_check=datetime.now(timezone.utc).isoformat()))
        return components
    
    def get_system_health(self) -> SystemHealth:
        components = [self.check_database(), self.check_vector_store(), self.check_model(), self.check_embeddings()]
        components.extend(self.check_circuit_breakers())
        statuses = [c.status for c in components]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        import os
        return SystemHealth(status=overall_status, timestamp=datetime.now(timezone.utc).isoformat(),
                            uptime_seconds=time.time() - self._start_time, components=components,
                            version=os.environ.get("CLDB_VERSION", "1.0.0"), environment=os.environ.get("CLDB_ENVIRONMENT", "development"))
    
    def get_prometheus_metrics(self) -> str:
        health = self.get_system_health()
        output = []
        status_value = 1.0 if health.status == HealthStatus.HEALTHY else 0.5 if health.status == HealthStatus.DEGRADED else 0.0
        output.append(f'cldb_health_status{{environment="{health.environment}"}} {status_value}')
        output.append(f'cldb_uptime_seconds{{environment="{health.environment}"}} {health.uptime_seconds}')
        for comp in health.components:
            status_value = 1.0 if comp.status == HealthStatus.HEALTHY else 0.5 if comp.status == HealthStatus.DEGRADED else 0.0
            output.append(f'cldb_component_health{{component="{comp.name}",environment="{health.environment}"}} {status_value}')
            if comp.latency_ms is not None:
                output.append(f'cldb_component_latency_ms{{component="{comp.name}"}} {comp.latency_ms}')
        return "\n".join(output)


class HealthServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, health_checker: Optional[HealthChecker] = None):
        self.host = host
        self.port = port
        self.health_checker = health_checker or HealthChecker()
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger = get_logger("health")
        logger.info(f"Health server started on {self.host}:{self.port}")
    
    def _run_server(self):
        import socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            while self._running:
                try:
                    client_socket, address = server_socket.accept()
                    self._handle_request(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger = get_logger("health")
                        logger.error(f"Server error: {e}")
        finally:
            server_socket.close()
    
    def _handle_request(self, client_socket):
        try:
            request = client_socket.recv(4096).decode('utf-8')
            if not request:
                return
            lines = request.split('\r\n')
            if not lines:
                return
            method, path, _ = lines[0].split(' ')
            response_body = ""
            status_code = 200
            
            if path == "/health" or path == "/health/json":
                import json
                health = self.health_checker.get_system_health()
                response_body = json.dumps({"status": health.status.value, "timestamp": health.timestamp, "uptime_seconds": health.uptime_seconds,
                                           "components": [{"name": c.name, "status": c.status.value, "latency_ms": c.latency_ms, "message": c.message} for c in health.components]}, indent=2)
                content_type = "application/json"
            elif path == "/health/text":
                health = self.health_checker.get_system_health()
                response_body = f"Status: {health.status.value}\nUptime: {health.uptime_seconds:.2f}s\n"
                for comp in health.components:
                    response_body += f"  {comp.name}: {comp.status.value}"
                    if comp.latency_ms:
                        response_body += f" ({comp.latency_ms:.2f}ms)"
                    response_body += "\n"
                content_type = "text/plain"
            elif path == "/metrics":
                response_body = self.health_checker.get_prometheus_metrics()
                content_type = "text/plain"
            elif path == "/ready":
                health = self.health_checker.get_system_health()
                response_body = "READY" if health.status == HealthStatus.HEALTHY else "NOT READY"
                status_code = 503 if health.status != HealthStatus.HEALTHY else 200
                content_type = "text/plain"
            elif path == "/live":
                response_body = "ALIVE"
                content_type = "text/plain"
            else:
                status_code = 404
                response_body = "Not Found"
                content_type = "text/plain"
            
            response = f"HTTP/1.1 {status_code} OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(response_body.encode('utf-8'))}\r\nConnection: close\r\n\r\n{response_body}"
            client_socket.send(response.encode('utf-8'))
        except Exception as e:
            logger = get_logger("health")
            logger.error(f"Request handling error: {e}")
        finally:
            client_socket.close()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None