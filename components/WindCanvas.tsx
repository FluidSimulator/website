'use client';

import { useRef, useEffect } from 'react';

export default function WindCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -1, y: -1, active: false });
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    // Delay sizing by one frame so the parent div has settled its layout
    const initId = requestAnimationFrame(() => {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      start();
    });

    function start() {
      const w = canvas!.width, h = canvas!.height;
      if (w < 10 || h < 10) return;

      const particles: any[] = [];
      for (let i = 0; i < 160; i++) {
        const a = Math.random() * Math.PI * 2, s = 0.5 + Math.random() * 2;
        particles.push({ x: Math.random() * w, y: Math.random() * h, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: 0, maxLife: 120 + Math.random() * 180, size: 1 + Math.random() * 2, history: [] as any[] });
      }

      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, w, h);

      const tick = () => {
        ctx.fillStyle = 'rgba(0,0,0,0.07)';
        ctx.fillRect(0, 0, w, h);
        const m = mouseRef.current;

        for (const p of particles) {
          p.life++;
          p.history.push({ x: p.x, y: p.y });
          if (p.history.length > 14) p.history.shift();
          const na = Math.sin(p.x * 0.005 + p.life * 0.02) * Math.cos(p.y * 0.005) * Math.PI;
          p.vx += Math.cos(na) * 0.08; p.vy += Math.sin(na) * 0.08;
          if (m.active && m.x > 0) {
            const dx = m.x - p.x, dy = m.y - p.y, dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 200) { const f = (200 - dist) / 200 * 0.35; p.vx += (-dy / dist) * f + (dx / dist) * f * 0.3; p.vy += (dx / dist) * f + (dy / dist) * f * 0.3; }
          }
          p.vx *= 0.97; p.vy *= 0.97;
          const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
          if (spd > 4) { p.vx = (p.vx / spd) * 4; p.vy = (p.vy / spd) * 4; }
          p.x += p.vx; p.y += p.vy;
          if (p.x < -10) p.x = w + 10; if (p.x > w + 10) p.x = -10;
          if (p.y < -10) p.y = h + 10; if (p.y > h + 10) p.y = -10;
          if (p.life > p.maxLife) { p.x = Math.random() * w; p.y = Math.random() * h; p.life = 0; p.history = []; }
          const alpha = Math.min(1, p.life / 20) * (1 - p.life / p.maxLife);
          if (p.history.length > 1) {
            ctx.beginPath(); ctx.moveTo(p.history[0].x, p.history[0].y);
            for (let j = 1; j < p.history.length; j++) ctx.lineTo(p.history[j].x, p.history[j].y);
            ctx.lineTo(p.x, p.y); ctx.strokeStyle = `rgba(255,255,255,${alpha * 0.5})`; ctx.lineWidth = p.size * 0.8; ctx.stroke();
          }
          ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fillStyle = `rgba(255,255,255,${alpha * 0.9})`; ctx.fill();
          ctx.beginPath(); ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2); ctx.fillStyle = `rgba(200,232,200,${alpha * 0.06})`; ctx.fill();
        }
        animRef.current = requestAnimationFrame(tick);
      };
      animRef.current = requestAnimationFrame(tick);
    }

    return () => { cancelAnimationFrame(initId); cancelAnimationFrame(animRef.current); };
  }, []);

  const setMouse = (e: React.PointerEvent, active: boolean) => {
    const r = canvasRef.current!.getBoundingClientRect();
    mouseRef.current = { x: e.clientX - r.left, y: e.clientY - r.top, active };
  };

  return (
    <div className="canvas-frame wind w-full mx-auto" style={{ maxWidth: 800, aspectRatio: '2 / 1' }}>
      <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair"
        onPointerDown={e => setMouse(e, true)} onPointerMove={e => { if (mouseRef.current.active) setMouse(e, true); }}
        onPointerUp={e => setMouse(e, false)} onPointerLeave={() => { mouseRef.current.active = false; }} />
    </div>
  );
}
