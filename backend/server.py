"""
Fluid Simulator WebSocket Backend
==================================
Spawns each Taichi simulator as a separate subprocess (sim_worker.py)
so each gets its own ti.init(). Bridges WebSocket ↔ subprocess I/O.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fluid Simulator Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKER_PATH = SCRIPT_DIR / "sim_worker.py"
_docker_path = Path("/simulators")
_local_path  = SCRIPT_DIR.parent / "simulators"
SIM_DIR = _docker_path if _docker_path.exists() else _local_path

REGIONS = {
    "wind": {"name": "Wind Tunnel", "repo": "Wind_Tunnel", "entry": "Wind_Tunnel.py"},
    "fire": {"name": "Fire Simulator", "repo": "Fire_sim", "entry": "main.py"},
    "water": {"name": "Water Simulator", "repo": "water_sim", "entry": "main.py"},
    "paint": {"name": "Paint Simulator", "repo": "Eulerian_paint_sim", "entry": "Eulerian_paint.py"},
}


def check_simulator(region: str) -> bool:
    """Check if the simulator source files exist."""
    info = REGIONS[region]
    repo_path = SIM_DIR / info["repo"]
    entry_file = repo_path / info["entry"]
    return entry_file.exists()


@app.websocket("/ws/{region}")
async def simulation_ws(websocket: WebSocket, region: str):
    """
    WebSocket endpoint. Spawns a sim_worker.py subprocess for the
    requested region and bridges frames/mouse events.

    sim_worker.py stdout → WebSocket (JSON frames with base64 PNG)
    WebSocket → sim_worker.py stdin (JSON mouse events)
    """
    if region not in REGIONS:
        await websocket.close(code=4000, reason=f"Unknown region: {region}")
        return

    await websocket.accept()

    # Check if simulator files exist
    if not check_simulator(region):
        await websocket.send_json({
            "type": "status",
            "message": f"{region} simulator files not found",
            "initialized": False,
        })
        # Keep connection alive but tell frontend to use JS fallback
        try:
            while True:
                await asyncio.sleep(60)
        except:
            return

    # Spawn the simulator subprocess
    proc = None
    try:
        await websocket.send_json({
            "type": "status",
            "message": f"Launching {region} Taichi simulator...",
            "initialized": True,
        })

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(WORKER_PATH), region,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SCRIPT_DIR),
        )

        # ── Task 1: Read frames from subprocess stdout → send to WebSocket ──
        async def forward_frames():
            while proc.returncode is None:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode().strip())
                    await websocket.send_json(msg)
                except (json.JSONDecodeError, RuntimeError):
                    continue

        # ── Task 2: Read mouse from WebSocket → send to subprocess stdin ──
        async def forward_mouse():
            while proc.returncode is None:
                try:
                    data = await websocket.receive_text()
                    # Forward the raw JSON to the subprocess
                    proc.stdin.write((data.strip() + "\n").encode())
                    await proc.stdin.drain()
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception:
                    break

        # ── Task 3: Log stderr for debugging ──
        async def log_stderr():
            while proc.returncode is None:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    print(f"  [{region}] {text}", file=sys.stderr)

        # Run all tasks concurrently, stop when any finishes
        tasks = [
            asyncio.create_task(forward_frames()),
            asyncio.create_task(forward_mouse()),
            asyncio.create_task(log_stderr()),
        ]

        # Wait for first task to complete (disconnect or process exit)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Kill the subprocess
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except:
                pass


@app.get("/")
async def root():
    return {
        "name": "Fluid Simulator Backend",
        "version": "2.0.0",
        "websocket": "ws://localhost:8000/ws/{region}",
        "regions": {
            k: {
                **v,
                "available": check_simulator(k),
                "path": str(SIM_DIR / v["repo"]),
            }
            for k, v in REGIONS.items()
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "simulators": {
            k: {
                "available": check_simulator(k),
                "path": str(SIM_DIR / v["repo"] / v["entry"]),
            }
            for k, v in REGIONS.items()
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    