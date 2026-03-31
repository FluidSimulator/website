# Interactive Pixel Art Website

Pokemon-themed adventure map integrated with **Taichi-based Python fluid simulators** via WebSocket. Click any region on the map to launch its interactive physics simulation.

## Architecture

```
┌──────────────── Next.js Frontend ─────────────────┐
│                                                    │
│  AdventureMap (Figma design parchment image)       │
│       ↓ click region                               │
│  RegionModal + SimulatorCanvas                     │
│    ├─ tries WebSocket ws://localhost:8000/ws/{r}    │
│    ├─ CONNECTED → renders Taichi backend frames    │
│    └─ OFFLINE   → renders JS fallback canvas       │
│                                                    │
└────────────────────┬───────────────────────────────┘
                     │ WebSocket (PNG frames ↑ mouse ↓)
┌────────────────────┴───────────────────────────────┐
│          Python FastAPI Backend (server.py)          │
│                                                     │
│  /ws/wind  → Wind_Tunnel/Wind_Tunnel.py             │
│     MAC staggered grid, Red-Black Gauss-Seidel,     │
│     semi-Lagrangian advection, CNN super-resolution  │
│                                                     │
│  /ws/fire  → Fire_sim/main.py + src/                │
│     Eulerian Stable Fluids, vorticity confinement,   │
│     temperature buoyancy, SOR-Jacobi, bloom          │
│                                                     │
│  /ws/water → water_sim/main.py + src/               │
│     FLIP/PIC hybrid, ML-accelerated pressure solver  │
│                                                     │
│  /ws/paint → Eulerian_paint_sim/main.py             │
│     Eulerian paint/color mixing simulation           │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Option A: Frontend Only (works immediately)

```bash
npm install
npm run dev
# Open http://localhost:3000
```

Runs with built-in JS particle simulations. No backend needed.

### Option B: Full Stack (real Taichi physics)

All four simulator repos are already included in `simulators/`.

```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt
pip install taichi torch numpy  # for the Taichi simulators

# 2. Start backend (auto-finds simulators/ folder)
uvicorn server:app --host 0.0.0.0 --port 8000

# 3. Start frontend (separate terminal)
cd ..
cp .env.example .env.local
npm install
npm run dev
```

### Option C: Docker Compose

```bash
# Clone all repos first (step 1 above), then:
docker compose up
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

## How It Works

### SimulatorCanvas (the key integration component)

When you click a region on the map, `SimulatorCanvas` does:

1. **Tries WebSocket** → `ws://localhost:8000/ws/{region}`
2. **If backend responds** → streams Taichi-rendered frames onto `<canvas>`, sends mouse input back. Green "TAICHI" badge appears.
3. **If backend offline** → renders the JS fallback component (`WindCanvas`, `FireCanvas`, `WaterCanvas`, `PaintCanvas`). Orange "JS FALLBACK" pill appears.

### Backend WebSocket Protocol

```
Server → Client:
  { "type": "frame", "data": "<base64 PNG>", "frame": 42 }
  { "type": "status", "message": "...", "initialized": true }

Client → Server:
  { "type": "mouse", "x": 0.5, "y": 0.3, "down": true }
  { "type": "reset" }
```

### Backend Simulator Integration

Each adapter in `server.py` wraps a specific Taichi repo:

| Adapter | Repo | Key Files | Integration Point |
|---|---|---|---|
| `WindAdapter` | `Wind_Tunnel` | `Wind_Tunnel.py` | `simulate()` + `k_neural_colormap()` → `pixels` field |
| `FireAdapter` | `Fire_sim` | `src/sim.py`, `src/renderer.py` | `simulation_step()` + `render()` → `pixels` field |
| `WaterAdapter` | `water_sim` | `main.py`, `src/` | FLIP step + render → pixel buffer |
| `PaintAdapter` | `Eulerian_paint_sim` | `main.py` | sim step + render → pixel buffer |

Each adapter:
1. Adds the repo to `sys.path`
2. Imports Taichi and initializes kernels
3. Per frame: calls `simulation_step()`, extracts pixel buffer via `.to_numpy()`, encodes as PNG, sends via WebSocket

## Project Structure

```
pixel-art-site/
├── app/
│   ├── globals.css            # Mossy bg, parchment, region styling, pokeball cursor
│   ├── layout.tsx             # Root layout with fonts
│   └── page.tsx               # Main orchestrator
├── components/
│   ├── AdventureMap.tsx        # Parchment map image with clickable region overlays
│   ├── RegionModal.tsx         # Modal overlay, banner, instruction, preview thumbs
│   ├── Sidebar.tsx             # Left nav icons to switch regions
│   ├── SimulatorCanvas.tsx     # ★ WebSocket bridge to Python backend + fallback
│   ├── WindCanvas.tsx          # JS fallback: wind particles
│   ├── FireCanvas.tsx          # JS fallback: fire embers
│   ├── WaterCanvas.tsx         # JS fallback: water droplets + ripples
│   └── PaintCanvas.tsx         # JS fallback: paint constellation
├── hooks/
│   └── useSimulator.ts        # WebSocket connection hook
├── backend/
│   ├── server.py               # FastAPI WebSocket server with simulator adapters
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile              # Backend container
├── simulators/                 # ★ ALL PYTHON TAICHI SIMULATORS (from GitHub)
│   ├── Wind_Tunnel/
│   │   └── Wind_Tunnel.py      # MAC grid NS + CNN super-res (27 KB)
│   ├── Fire_sim/
│   │   ├── main.py             # Entry point
│   │   └── src/                # config, fields, kernels, renderer, sim, ML
│   ├── water_sim/
│   │   ├── main.py             # Entry point
│   │   └── src/                # config, sim, ml_solver
│   └── Eulerian_paint_sim/
│       └── Eulerian_paint.py   # Eulerian paint sim (31 KB)
├── public/
│   ├── map.png                 # Adventure map (from Figma design)
│   ├── wind-preview.png        # Region preview thumbnails
│   ├── fire-preview.png
│   ├── water-preview.png
│   └── paint-preview.png
├── docker-compose.yml          # One-command full-stack startup
├── Dockerfile.frontend         # Frontend container
├── .env.example                # Backend URL config
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── postcss.config.js
```

## Backend Simulators

| Region | Physics | FPS | Special |
|---|---|---|---|
| Wind | Incompressible Navier-Stokes (MAC grid) | ~34 | CNN super-resolution (PyTorch), warm-started pressure, async pipelined inference |
| Fire | Eulerian Stable Fluids | ~65 | Vorticity confinement, temperature buoyancy, cinematic color ramp, bloom |
| Water | FLIP/PIC hybrid | ~30 | Neural network pressure predictor trained from simulation data |
| Paint | Eulerian fluid | ~40 | Color mixing, paint dynamics |

All run on **CPU only** via Taichi parallel kernels. No GPU required.
