'use client';

import { useRef, useEffect } from 'react';

export default function WaterCanvas() {
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

      const drops: any[] = [];
      const ripples: any[] = [];
      function mk() { return { x: Math.random() * w, y: -10 - Math.random() * 80, vy: 2 + Math.random() * 3, vx: (Math.random() - 0.5) * 0.3, size: 1.5 + Math.random() * 2.5, life: 0, maxLife: 150 + Math.random() * 100, curve: (Math.random() - 0.5) * 0.02, trail: [] as any[] }; }
      for (let i = 0; i < 80; i++) drops.push(mk());
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);

      const tick = () => {
        ctx.fillStyle = 'rgba(0,0,0,0.1)'; ctx.fillRect(0, 0, w, h);
        const m = mouseRef.current;
        if (drops.length < 100 && Math.random() < 0.3) drops.push(mk());
        if (m.active && m.x > 0 && Math.random() < 0.15) ripples.push({ x: m.x + (Math.random() - 0.5) * 20, y: m.y + (Math.random() - 0.5) * 20, r: 2, mr: 40 + Math.random() * 30 });

        for (let i = ripples.length - 1; i >= 0; i--) {
          const rr = ripples[i]; rr.r += 2;
          const a = Math.max(0, 1 - rr.r / rr.mr);
          ctx.beginPath(); ctx.arc(rr.x, rr.y, rr.r, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(0,200,255,${a * 0.3})`; ctx.lineWidth = 2; ctx.stroke();
          if (rr.r > rr.mr) ripples.splice(i, 1);
        }

        for (let i = drops.length - 1; i >= 0; i--) {
          const d = drops[i]; d.life++;
          d.trail.push({ x: d.x, y: d.y }); if (d.trail.length > 8) d.trail.shift();
          d.vx += d.curve; d.x += d.vx; d.y += d.vy;
          if (m.active && m.x > 0) {
            const dx = d.x - m.x, dy = d.y - m.y, dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 120) { const f = (120 - dist) / 120; d.vx += (dx / dist) * f * 0.5; d.vy += (dy / dist) * f * 0.3; }
          }
          if (d.y > h + 10 || d.x < -10 || d.x > w + 10 || d.life > d.maxLife) {
            if (d.y > h - 20) ripples.push({ x: d.x, y: h - 5, r: 2, mr: 20 + Math.random() * 15 });
            drops.splice(i, 1); continue;
          }
          const alpha = Math.min(1, d.life / 8) * Math.max(0.2, 1 - d.life / d.maxLife);
          if (d.trail.length > 1) {
            ctx.beginPath(); ctx.moveTo(d.trail[0].x, d.trail[0].y);
            for (let j = 1; j < d.trail.length; j++) ctx.lineTo(d.trail[j].x, d.trail[j].y);
            ctx.lineTo(d.x, d.y); ctx.strokeStyle = `rgba(0,180,255,${alpha * 0.4})`; ctx.lineWidth = d.size * 0.6; ctx.lineCap = 'round'; ctx.stroke();
          }
          ctx.beginPath(); ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2); ctx.fillStyle = `rgba(0,200,255,${alpha})`; ctx.fill();
          ctx.beginPath(); ctx.arc(d.x, d.y, d.size * 0.4, 0, Math.PI * 2); ctx.fillStyle = `rgba(150,240,255,${alpha * 0.9})`; ctx.fill();
          ctx.beginPath(); ctx.arc(d.x, d.y, d.size * 4, 0, Math.PI * 2); ctx.fillStyle = `rgba(0,150,220,${alpha * 0.06})`; ctx.fill();
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
    <div className="canvas-frame water w-full mx-auto" style={{ maxWidth: 600, aspectRatio: '4 / 3' }}>
      <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair"
        onPointerDown={e => setMouse(e, true)} onPointerMove={e => { if (mouseRef.current.active) setMouse(e, true); }}
        onPointerUp={e => setMouse(e, false)} onPointerLeave={() => { mouseRef.current.active = false; }} />
    </div>
  );
}
