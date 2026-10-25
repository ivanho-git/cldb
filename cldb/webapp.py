"""CLDB live web application.

A FastAPI server that hosts a single-page control room for the heavy-traffic
simulator. The browser receives live snapshots over a WebSocket (with an HTTP
polling fallback) and controls the simulation with the same channel.

Run:
    python -m cldb.webapp --host 127.0.0.1 --port 8080 --qps 60

Then open http://127.0.0.1:8080.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cldb.logger import get_logger
from cldb.traffic import HeavyTrafficSimulator, TrafficConfig

WEB_DIR = Path(__file__).resolve().parent / "web"
logger = get_logger("webapp")

CONTROL_ACTIONS = {"start", "pause", "resume", "stop", "reset", "configure"}


def _apply_control(sim: HeavyTrafficSimulator, payload: dict) -> dict:
    """Apply a control message; never let a bad payload kill the app."""
    action = str(payload.get("action", "configure")).lower()
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    if config:
        sim.configure(**config)
    if action == "start":
        sim.start()
    elif action == "pause":
        sim.pause()
    elif action == "resume":
        sim.resume()
    elif action == "stop":
        sim.stop()
    elif action == "reset":
        sim.reset(seed=config.get("seed"))
    elif action not in CONTROL_ACTIONS:
        return {"ok": False, "error": f"unknown action: {action}"}
    return {"ok": True, "state": sim.snapshot()}


def create_app(simulator: HeavyTrafficSimulator) -> FastAPI:
    app = FastAPI(title="CLDB Live Index Advisor", version="2.0.0")
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/state")
    def state():
        return JSONResponse(simulator.snapshot())

    @app.get("/api/health")
    def health():
        snap = simulator.snapshot()
        return {
            "status": "ok" if snap["status"]["error"] is None else "error",
            "running": snap["status"]["running"],
            "paused": snap["status"]["paused"],
            "qps_target": snap["traffic"]["qps_target"],
            "qps_actual": snap["traffic"]["qps_actual"],
            "queries": snap["traffic"]["total_queries"],
            "phase": snap["phase"]["name"],
            "error": snap["status"]["error"],
        }

    @app.get("/api/benchmark")
    def benchmark():
        root = Path(__file__).resolve().parent.parent
        out = {}
        for name, rel in (
            ("continual", "experiments/results/cl_results.json"),
            ("cwil", "experiments/results/cwil_ablation.json"),
        ):
            path = root / rel
            if path.exists():
                try:
                    out[name] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    out[name] = None
            else:
                out[name] = None
        return out

    @app.post("/api/sim/control")
    async def control(payload: dict):
        return _apply_control(simulator, payload)

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket):
        await websocket.accept()
        try:
            await websocket.send_json(simulator.snapshot())
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = {}
                    if isinstance(payload, dict):
                        _apply_control(simulator, payload)
                except asyncio.TimeoutError:
                    pass
                await websocket.send_json(simulator.snapshot())
        except WebSocketDisconnect:
            return
        except Exception as exc:
            logger.warning(f"live socket closed: {exc}")
            try:
                await websocket.close()
            except Exception:
                pass

    return app


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="CLDB live web application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--qps", type=float, default=60.0)
    parser.add_argument("--steps-per-phase", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Epsilon floor for greedy evaluation.")
    parser.add_argument("--reheat-epsilon", type=float, default=0.35,
                        help="Epsilon restored after a phase change or drift (0 disables).")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Do not start traffic until the Start button is pressed.")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args(argv)

    config = TrafficConfig(qps=args.qps, steps_per_phase=args.steps_per_phase,
                           epsilon_end=args.epsilon_end,
                           reheat_epsilon=args.reheat_epsilon,
                           seed=args.seed).clamped()
    simulator = HeavyTrafficSimulator(config)
    if not args.no_autostart:
        simulator.start()

    app = create_app(simulator)
    url = f"http://{args.host}:{args.port}"
    logger.info(f"CLDB live app listening on {url}")
    print(f"\n  CLDB live app -> {url}\n"
          f"  traffic target: {config.qps:.0f} q/s   phase length: {config.steps_per_phase} steps\n")

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        simulator.stop()


if __name__ == "__main__":
    main()
