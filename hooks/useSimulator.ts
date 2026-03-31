'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

export type RegionName = 'wind' | 'fire' | 'water' | 'paint';

export interface MouseState {
  x: number;
  y: number;
  active: boolean;
}

/**
 * Hook that manages a WebSocket connection to the Python Taichi backend.
 * Used internally by SimulatorCanvas.
 *
 * Backend repos:
 *   Wind  → FluidSimulator/Wind_Tunnel   (MAC grid NS + CNN super-res)
 *   Fire  → FluidSimulator/Fire_sim      (Eulerian fire + vorticity)
 *   Water → FluidSimulator/water_sim     (FLIP/PIC + ML pressure)
 *   Paint → FluidSimulator/Eulerian_paint_sim
 */
export function useSimulator(region: RegionName, canvasRef: React.RefObject<HTMLCanvasElement>) {
  const wsRef = useRef<WebSocket | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [connected, setConnected] = useState(false);

  const wsUrl = process.env.NEXT_PUBLIC_BACKEND_WS || 'ws://localhost:8000';

  useEffect(() => {
    const url = `${wsUrl}/ws/${region}`;
    let didConnect = false;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        didConnect = true;
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'frame' && msg.data) {
            renderFrame(msg.data);
          }
        } catch {}
      };

      ws.onerror = () => { if (!didConnect) setConnected(false); };
      ws.onclose = () => setConnected(false);
    } catch {
      setConnected(false);
    }

    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [region, wsUrl]);

  const renderFrame = useCallback((base64Data: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (!imgRef.current) imgRef.current = new Image();
    imgRef.current.onload = () => ctx.drawImage(imgRef.current!, 0, 0, canvas.width, canvas.height);
    imgRef.current.src = `data:image/png;base64,${base64Data}`;
  }, [canvasRef]);

  const sendMouse = useCallback((x: number, y: number, down: boolean) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'mouse', x, y, down }));
    }
  }, []);

  return { connected, sendMouse };
}
