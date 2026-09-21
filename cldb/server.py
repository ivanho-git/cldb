"""
HTTP query-ingestion service for CLDB.

Exposes the continual-learning pipeline as a live service so query traffic can
be streamed in, learned from, and optimized against automatically.
"""
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict
from urllib.parse import urlparse

from cldb.logger import get_logger
from cldb.metrics import get_prometheus_metrics


class CLDBRequestHandler(BaseHTTPRequestHandler):
    server_version = "CLDB/1.0"

    @property
    def cl_server(self):
        return self.server.cl_server

    def _send_json(self, status: int, payload: Dict):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, body: str, content_type: str = "text/plain"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _health_payload(self) -> Dict:
        health = self.cl_server.health_checker.get_system_health()
        return {
            "status": health.status.value,
            "timestamp": health.timestamp,
            "uptime_seconds": health.uptime_seconds,
            "version": health.version,
            "environment": health.environment,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "last_check": c.last_check,
                }
                for c in health.components
            ],
        }

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/health/json"):
            self._send_json(200, self._health_payload())
        elif path == "/health/text":
            health = self.cl_server.health_checker.get_system_health()
            self._send_text(200, f"Status: {health.status.value}\n")
        elif path == "/ready":
            health = self.cl_server.health_checker.get_system_health()
            ready = health.status.value != "unhealthy"
            self._send_text(200 if ready else 503, "READY" if ready else "NOT READY")
        elif path == "/live":
            self._send_text(200, "ALIVE")
        elif path == "/metrics":
            self._send_text(200, get_prometheus_metrics())
        elif path == "/status":
            self._send_json(200, self.cl_server.pipeline.get_metrics_summary())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_body()
        except Exception as e:
            self._send_json(400, {"error": f"invalid JSON body: {e}"})
            return

        if path == "/query":
            query = body.get("query")
            if not query:
                self._send_json(400, {"error": "missing 'query' field"})
                return
            result = self.cl_server.pipeline.process_query(query)
            self._send_json(200, result)
        elif path == "/batch":
            queries = body.get("queries")
            if not isinstance(queries, list) or not queries:
                self._send_json(400, {"error": "missing 'queries' list"})
                return
            results = self.cl_server.pipeline.process_batch(queries)
            self._send_json(200, {"results": results})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        get_logger("http").info(f"{self.address_string()} - {format % args}")


class CLDBServer:
    def __init__(self, pipeline, health_checker, host: str = "0.0.0.0", port: int = 5000):
        self.pipeline = pipeline
        self.health_checker = health_checker
        self.host = host
        self.port = port
        self.logger = get_logger("server")
        self._thread: Optional[threading.Thread] = None
        self._httpd: Optional[ThreadingHTTPServer] = None

    def start(self):
        if self._thread:
            return

        self._httpd = ThreadingHTTPServer((self.host, self.port), CLDBRequestHandler)
        self._httpd.cl_server = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.logger.info(f"CLDB query-ingestion server listening on {self.host}:{self.port}")

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


def serve():
    from cldb.config import get_config
    from cldb.metadata_repo import MetadataRepo
    from cldb.enhanced_vector_store import VectorStore
    from cldb.production_pipeline import ProductionPipeline
    from cldb.health import HealthChecker

    config = get_config()
    logger = get_logger("main")

    repo = MetadataRepo()
    vector_store = VectorStore()
    pipeline = ProductionPipeline(
        repo=repo,
        vector_store=vector_store,
        confidence_threshold=config.safety.confidence_threshold,
    )

    health_checker = HealthChecker(
        repo=repo,
        vector_store=vector_store,
        cl_engine=pipeline.cl_engine,
        embedder=pipeline.embedder,
    )

    server = CLDBServer(
        pipeline=pipeline,
        health_checker=health_checker,
        host=config.server.host,
        port=config.server.port,
    )
    server.start()

    def _shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        server.stop()
        pipeline.online_scheduler.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)
