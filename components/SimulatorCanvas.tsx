'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import WindCanvas from './WindCanvas';
import FireCanvas from './FireCanvas';
import WaterCanvas from './WaterCanvas';
import PaintCanvas from './PaintCanvas';

type RegionName = 'wind' | 'fire' | 'water' | 'paint';

interface Props {
  region: RegionName;
}

const FALLBACKS: Record<RegionName, React.ComponentType> = {
  wind: WindCanvas, fire: FireCanvas, water: WaterCanvas, paint: PaintCanvas,
};

// Aspect ratios matching actual Taichi pixel fields
const STYLES: Record<RegionName, { cls: string; maxW: number; aspect: string }> = {
  wind:  { cls: 'wind',  maxW: 800, aspect: '2 / 1'   },  // 1000×500
  fire:  { cls: 'fire',  maxW: 640, aspect: '1 / 2'    },  // 128×256 (portrait!)
  water: { cls: 'water', maxW: 600, aspect: '1 / 1'    },  // 800×800
  paint: { cls: 'paint', maxW: 800, aspect: '8 / 5'    },  // 800×500
};

export default function SimulatorCanvas({ region }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [gotFrame, setGotFrame] = useState(false);
  const [status, setStatus] = useState('');

  const st = STYLES[region];
  const Fallback = FALLBACKS[region];

  // ── Try connecting to Taichi backend ──
  useEffect(() => {
    const url = `${process.env.NEXT_PUBLIC_BACKEND_WS || 'ws://localhost:8000'}/ws/${region}`;
    let alive = true;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        if (!alive) return;
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'frame' && msg.data) {
            if (!gotFrame) setGotFrame(true);
            drawFrame(msg.data);
          } else if (msg.type === 'status') {
            setStatus(msg.message || '');
            if (msg.initialized === false) ws.close();
          } else if (msg.type === 'error') {
            ws.close();
          }
        } catch {}
      };
      ws.onerror = () => ws.close();
      ws.onclose = () => { wsRef.current = null; };
    } catch {}

    // Clear stale status after 12s
    const t = setTimeout(() => { if (alive && !gotFrame) setStatus(''); }, 12000);

    return () => { alive = false; clearTimeout(t); wsRef.current?.close(); wsRef.current = null; };
  }, [region]);

  const drawFrame = useCallback((b64: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (canvas.width < 10) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    if (!imgRef.current) imgRef.current = new Image();
    imgRef.current.onload = () => ctx.drawImage(imgRef.current!, 0, 0, canvas.width, canvas.height);
    imgRef.current.src = `data:image/png;base64,${b64}`;
  }, []);

  const sendMouse = useCallback((down: boolean, e: React.PointerEvent) => {
    const ws = wsRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    ws.send(JSON.stringify({
      type: 'mouse',
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
      down,
    }));
  }, []);

  // ── NO backend frames yet → render fallback canvas DIRECTLY (no wrapper div) ──
  if (!gotFrame) {
    return <Fallback />;
  }

  // ── Backend frames flowing → render on our canvas ──
  return (
    <div
      className={`canvas-frame ${st.cls} mx-auto relative`}
      style={{ width: '100%', maxWidth: st.maxW, aspectRatio: st.aspect }}
    >
      <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 bg-black/50 backdrop-blur-sm px-2.5 py-1 rounded-md pointer-events-none">
        <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        <span className="text-[8px] text-white/60 font-silk tracking-wider">TAICHI</span>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full h-full block cursor-crosshair"
        onPointerDown={e => sendMouse(true, e)}
        onPointerMove={e => sendMouse(true, e)}
        onPointerUp={e => sendMouse(false, e)}
        onPointerLeave={e => sendMouse(false, e)}
      />
    </div>
  );
}
