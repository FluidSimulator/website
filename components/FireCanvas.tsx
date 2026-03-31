'use client';

import { useRef, useEffect } from 'react';

export default function FireCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -1, y: -1, active: false });
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    const initId = requestAnimationFrame(() => {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      start();
    });

    function start() {
      const w = canvas!.width, h = canvas!.height;
      if (w < 10 || h < 10) return;

      const embers: any[] = [];
      function mk(top: boolean) {
        const hue = Math.random() < 0.15 ? 50 + Math.random() * 10 : Math.random() * 30;
        return { x: Math.random() * w, y: top ? -10 - Math.random() * 50 : Math.random() * h * 0.3, vy: 1 + Math.random() * 3, vx: (Math.random() - 0.5) * 0.5, size: 1 + Math.random() * 3, life: 0, maxLife: 100 + Math.random() * 200, hue, bri: 50 + Math.random() * 50, trail: [] as any[] };
      }
      for (let i = 0; i < 130; i++) embers.push(mk(false));
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);

      const tick = () => {
        ctx.fillStyle = 'rgba(0,0,0,0.11)'; ctx.fillRect(0, 0, w, h);
        const m = mouseRef.current;
        if (embers.length < 190 && Math.random() < 0.4) embers.push(mk(true));
        for (let i = embers.length - 1; i >= 0; i--) {
          const e = embers[i]; e.life++;
          e.trail.push({ x: e.x, y: e.y }); if (e.trail.length > 22) e.trail.shift();
          e.vy += 0.02; e.vx += (Math.random() - 0.5) * 0.08; e.vx *= 0.99;
          if (m.active && m.x > 0) {
            const dx = e.x - m.x, dy = e.y - m.y, dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150) { const f = (150 - dist) / 150; e.vx += (dx / dist) * f * 0.8; e.vy += (dy / dist) * f * 0.5; e.bri = Math.min(100, e.bri + 5); }
          }
          e.x += e.vx; e.y += e.vy;
          if (e.y > h + 20 || e.life > e.maxLife || e.x < -20 || e.x > w + 20) { embers.splice(i, 1); continue; }
          const alpha = Math.min(1, e.life / 10) * Math.max(0, 1 - e.life / e.maxLife);
          if (e.trail.length > 1) {
            ctx.beginPath(); ctx.moveTo(e.trail[0].x, e.trail[0].y);
            for (let j = 1; j < e.trail.length; j++) ctx.lineTo(e.trail[j].x, e.trail[j].y);
            ctx.strokeStyle = `hsla(${e.hue},100%,${e.bri * 0.6}%,${alpha * 0.4})`; ctx.lineWidth = e.size * 0.6; ctx.lineCap = 'round'; ctx.stroke();
          }
          ctx.beginPath(); ctx.arc(e.x, e.y, e.size, 0, Math.PI * 2); ctx.fillStyle = `hsla(${e.hue},100%,${e.bri}%,${alpha})`; ctx.fill();
          ctx.beginPath(); ctx.arc(e.x, e.y, e.size * 0.5, 0, Math.PI * 2); ctx.fillStyle = `hsla(${e.hue + 10},100%,90%,${alpha * 0.8})`; ctx.fill();
          ctx.beginPath(); ctx.arc(e.x, e.y, e.size * 4, 0, Math.PI * 2); ctx.fillStyle = `hsla(${e.hue},100%,${e.bri * 0.4}%,${alpha * 0.06})`; ctx.fill();
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
    <div className="canvas-frame fire w-full mx-auto" style={{ maxWidth: 800, aspectRatio: '16 / 10' }}>
      <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair"
        onPointerDown={e => setMouse(e, true)} onPointerMove={e => { if (mouseRef.current.active) setMouse(e, true); }}
        onPointerUp={e => setMouse(e, false)} onPointerLeave={() => { mouseRef.current.active = false; }} />
    </div>
  );
}
