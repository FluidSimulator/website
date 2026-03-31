#!/usr/bin/env python3
"""
sim_worker.py — Headless Taichi simulator subprocess
=====================================================
Spawned by server.py for each WebSocket connection.
Each process gets its own ti.init() and runs one simulator.

Usage:  python sim_worker.py <region>

Protocol (JSON lines):
  stdout → {"type":"frame","data":"<base64 PNG>","frame":N}
  stdin  ← {"type":"mouse","x":0.5,"y":0.3,"down":true}
"""

import sys, os, json, time, io, base64, threading
import numpy as np

REGION = sys.argv[1] if len(sys.argv) > 1 else "fire"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(SCRIPT_DIR, "..", "simulators")

# Shared mouse state (written by stdin reader thread)
mouse_x = 0.5
mouse_y = 0.5
mouse_down = False
should_reset = False


def stdin_reader():
    """Background thread reading mouse events from stdin."""
    global mouse_x, mouse_y, mouse_down, should_reset
    for line in sys.stdin:
        try:
            msg = json.loads(line.strip())
            if msg.get("type") == "mouse":
                mouse_x = msg.get("x", 0.5)
                mouse_y = msg.get("y", 0.5)
                mouse_down = msg.get("down", False)
            elif msg.get("type") == "reset":
                should_reset = True
        except:
            pass


def encode_frame(pixels_np, target_w, target_h):
    """
    Convert Taichi pixel field numpy array → base64 PNG.
    Taichi fields are (W, H, 3) float32 [0,1], origin at bottom-left.
    PIL expects (H, W, 3) uint8, origin at top-left.
    """
    from PIL import Image

    arr = np.clip(pixels_np, 0.0, 1.0)
    # (W, H, 3) → (H, W, 3), flip Y
    arr = np.transpose(arr, (1, 0, 2))[::-1]
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr, 'RGB')
    img = img.resize((target_w, target_h), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=1)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIRE  —  FluidSimulator/Fire_sim
#  Eulerian Stable Fluids, MAC grid, vorticity confinement, SOR-Jacobi
#  Entry: main.py → src/sim.py simulation_step() → src/fields.py pixels
# ═══════════════════════════════════════════════════════════════════════════════

def run_fire():
    global should_reset
    sys.path.insert(0, os.path.join(SIM_DIR, "Fire_sim"))

    import taichi as ti
    ti.init(arch=ti.cpu, cpu_max_num_threads=8, default_fp=ti.f32,
            fast_math=True, debug=False)

    from src.config import GRID_W, GRID_H, OBSTACLE_RADIUS
    from src.fields import pixels                       # ti.Vector.field(3, f32, (128,256))
    from src.kernels import reset_all_fields
    from src.sim import simulation_step, init_obstacle, move_obstacle_to

    reset_all_fields()
    init_obstacle()

    frame = 0
    dt = 1.0 / 25

    while True:
        t0 = time.time()

        if should_reset:
            reset_all_fields(); init_obstacle(); frame = 0; should_reset = False

        if mouse_down:
            cx = int(mouse_x * GRID_W)
            cy = int((1.0 - mouse_y) * GRID_H)
            move_obstacle_to(cx, cy, OBSTACLE_RADIUS)

        simulation_step()                               # runs all NS + render_pixels()
        frame += 1

        b64 = encode_frame(pixels.to_numpy(), 400, 250) # (128,256,3) → 400×250 PNG
        send({"type": "frame", "data": b64, "frame": frame})

        sl = dt - (time.time() - t0)
        if sl > 0: time.sleep(sl)


# ═══════════════════════════════════════════════════════════════════════════════
#  WIND  —  FluidSimulator/Wind_Tunnel
#  MAC staggered grid, Red-Black Gauss-Seidel, CNN super-resolution
#  Single file: Wind_Tunnel.py with k_init(), simulate(), render_preview()
# ═══════════════════════════════════════════════════════════════════════════════

def run_wind():
    global should_reset
    sys.path.insert(0, os.path.join(SIM_DIR, "Wind_Tunnel"))

    # Wind_Tunnel.py calls ti.init() at module level — do NOT call it here
    import importlib
    wt = importlib.import_module("Wind_Tunnel")

    # wt.k_init() sets up fields, obstacle, smoke
    wt.k_init()

    frame = 0
    dt = 1.0 / 20

    while True:
        t0 = time.time()

        if should_reset:
            wt.k_init(); frame = 0; should_reset = False

        if mouse_down:
            cx = int(mouse_x * wt.NX)
            cy = int((1.0 - mouse_y) * wt.NY)
            cx = max(wt.OBS_R + 1, min(cx, wt.NX - wt.OBS_R - 1))
            cy = max(wt.OBS_R + 1, min(cy, wt.NY - wt.OBS_R - 1))
            wt.k_move_obstacle(cx, cy)

        wt.simulate()                                   # NS step
        wt.render_preview()                              # smoke → sr_field → pixels
        frame += 1

        b64 = encode_frame(wt.pixels.to_numpy(), 500, 250)  # (1000,500,3) → 500×250
        send({"type": "frame", "data": b64, "frame": frame})

        sl = dt - (time.time() - t0)
        if sl > 0: time.sleep(sl)


# ═══════════════════════════════════════════════════════════════════════════════
#  WATER  —  FluidSimulator/water_sim
#  FLIP/PIC hybrid, ML-accelerated pressure solver
#  Entry: main.py → src/sim.py FlipSimulation.step(), render_frame(), pixels
# ═══════════════════════════════════════════════════════════════════════════════

def run_water():
    global should_reset
    sys.path.insert(0, os.path.join(SIM_DIR, "water_sim"))

    import taichi as ti
    ti.init(arch=ti.cpu, cpu_max_num_threads=8, default_fp=ti.f32,
            fast_math=True, debug=False)

    from src.config import WINDOW_W, WINDOW_H, SIM_WIDTH, SIM_HEIGHT
    from src.sim import (
        FlipSimulation, init_solid, init_particles, init_obstacle,
        render_frame, pixels, obs_x, obs_y, obs_vx, obs_vy,
    )

    init_solid()
    init_particles()
    init_obstacle()
    sim = FlipSimulation()

    frame = 0
    dt = 1.0 / 20

    while True:
        t0 = time.time()

        if should_reset:
            init_solid(); init_particles(); init_obstacle()
            sim = FlipSimulation()
            frame = 0; should_reset = False

        if mouse_down:
            nx = mouse_x * SIM_WIDTH
            ny = (1.0 - mouse_y) * SIM_HEIGHT
            obs_vx[None] = (nx - obs_x[None]) * 30.0
            obs_vy[None] = (ny - obs_y[None]) * 30.0
            obs_x[None] = nx
            obs_y[None] = ny
        else:
            obs_vx[None] = 0.0
            obs_vy[None] = 0.0

        sim.step()
        render_frame(0)                                  # 0 = no grid overlay
        frame += 1

        b64 = encode_frame(pixels.to_numpy(), 400, 400)  # (800,800,3) → 400×400
        send({"type": "frame", "data": b64, "frame": frame})

        sl = dt - (time.time() - t0)
        if sl > 0: time.sleep(sl)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAINT  —  FluidSimulator/Eulerian_paint_sim
#  Eulerian fluid with ML pressure warm-start, paint mixing
#  Single file: Eulerian_paint.py with MLAccelerator, render(), img field
# ═══════════════════════════════════════════════════════════════════════════════

def run_paint():
    global should_reset
    sys.path.insert(0, os.path.join(SIM_DIR, "Eulerian_paint_sim"))

    # Eulerian_paint.py calls ti.init() at module level — do NOT call it here
    import importlib
    ep = importlib.import_module("Eulerian_paint")

    # Initialise simulation state
    ep.reset_sim()
    ml = ep.MLAccelerator()
    ml._s_dirty = True

    frame = 0
    dt = 1.0 / 25

    while True:
        t0 = time.time()

        if should_reset:
            ep.reset_sim()
            ml = ep.MLAccelerator()
            ml._s_dirty = True
            frame = 0; should_reset = False

        if mouse_down:
            world_mx = mouse_x * ep.NX * ep.H
            world_my = (1.0 - mouse_y) * ep.NY * ep.H
            r = ep.obs_r[None]
            new_cx = float(np.clip(world_mx, r + 2*ep.H, ep.NX*ep.H - r - 2*ep.H))
            new_cy = float(np.clip(world_my, r + 2*ep.H, ep.NY*ep.H - r - 2*ep.H))
            ep.obs_vx[None] = (new_cx - ep.obs_cx[None]) / ep.SUB_DT
            ep.obs_vy[None] = (new_cy - ep.obs_cy[None]) / ep.SUB_DT
            ep.obs_cx[None] = new_cx
            ep.obs_cy[None] = new_cy
            ep.rebuild_solid()
            ml._s_dirty = True
            ep.p.fill(0.0)
            ep.p_disp.fill(0.0)
        else:
            ep.obs_vx[None] = 0.0
            ep.obs_vy[None] = 0.0

        # Physics substeps
        for _ in range(ep.SUBSTEPS):
            ml.step(ep.SUB_DT, False, is_dragging=mouse_down)

        # ML phase transitions
        frame += 1
        if ml.phase == ep.MLAccelerator.PHASE_COLLECT:
            ml.frame_count += 1
            if ml.frame_count >= ep.COLLECT_FRAMES:
                ml.phase = ep.MLAccelerator.PHASE_TRAIN
                ml.train()

        # Render
        ep.update_p_display(0.7)
        ep.render(0)                                     # 0 = paint view (not pressure)

        b64 = encode_frame(ep.img.to_numpy(), 400, 250)  # (800,500,3) → 400×250
        send({"type": "frame", "data": b64, "frame": frame})

        sl = dt - (time.time() - t0)
        if sl > 0: time.sleep(sl)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t = threading.Thread(target=stdin_reader, daemon=True)
    t.start()

    runners = {"fire": run_fire, "wind": run_wind, "water": run_water, "paint": run_paint}

    if REGION not in runners:
        send({"type": "error", "message": f"Unknown region: {REGION}"})
        sys.exit(1)

    try:
        send({"type": "status", "message": f"Starting {REGION} Taichi simulator...", "initialized": True})
        runners[REGION]()
    except Exception as e:
        send({"type": "error", "message": f"Simulator crashed: {e}"})
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
