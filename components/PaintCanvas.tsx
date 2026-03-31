'use client';

import { useRef, useEffect } from 'react';

const PC = ['#ff006e','#ff3388','#ff69b4','#cc44ff','#9933ff','#bb66ff','#ffcc00','#ffd700','#ffaa00','#ff4444','#44ff44'];

export default function PaintCanvas() {
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
      const rc = () => PC[Math.floor(Math.random() * PC.length)];

      const dots: any[] = [];
      for (let i = 0; i < 45; i++) {
        const a = Math.random() * Math.PI * 2, s = 0.2 + Math.random() * 0.5;
        dots.push({ x: Math.random() * w, y: Math.random() * h, vx: Math.cos(a) * s, vy: Math.sin(a) * s, size: 3 + Math.random() * 5, color: rc(), life: 0, maxLife: 300 + Math.random() * 400 });
      }
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);

      const tick = () => {
        ctx.fillStyle = 'rgba(0,0,0,0.04)'; ctx.fillRect(0, 0, w, h);
        const m = mouseRef.current;
        if (m.active && m.x > 0 && Math.random() < 0.35) {
          const a = Math.random() * Math.PI * 2, s = 1 + Math.random() * 3;
          dots.push({ x: m.x + (Math.random() - 0.5) * 30, y: m.y + (Math.random() - 0.5) * 30, vx: Math.cos(a) * s, vy: Math.sin(a) * s, size: 3 + Math.random() * 5, color: rc(), life: 0, maxLife: 300 + Math.random() * 400 });
        }
        while (dots.length > 220) dots.shift();
        for (let i = dots.length - 1; i >= 0; i--) {
          const d = dots[i]; d.life++; d.vx *= 0.995; d.vy *= 0.995; d.vy += 0.003;
          d.x += d.vx; d.y += d.vy;
          if (d.x < 5) { d.x = 5; d.vx *= -0.5; } if (d.x > w - 5) { d.x = w - 5; d.vx *= -0.5; }
          if (d.y < 5) { d.y = 5; d.vy *= -0.5; } if (d.y > h - 5) { d.y = h - 5; d.vy *= -0.5; }
          if (d.life > d.maxLife) dots.splice(i, 1);
        }
        const cd = 80;
        for (let i = 0; i < dots.length; i++) for (let j = i + 1; j < dots.length; j++) {
          const dx = dots[i].x - dots[j].x, dy = dots[i].y - dots[j].y, dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < cd) {
            const aI = Math.min(1, dots[i].life / 10) * Math.max(0.1, 1 - dots[i].life / dots[i].maxLife);
            const aJ = Math.min(1, dots[j].life / 10) * Math.max(0.1, 1 - dots[j].life / dots[j].maxLife);
            ctx.beginPath(); ctx.moveTo(dots[i].x, dots[i].y); ctx.lineTo(dots[j].x, dots[j].y);
            ctx.strokeStyle = `rgba(180,180,180,${Math.min(aI, aJ) * (1 - dist / cd) * 0.4})`; ctx.lineWidth = 0.8; ctx.stroke();
          }
        }
        for (const d of dots) {
          const alpha = Math.min(1, d.life / 8) * Math.max(0.15, 1 - d.life / d.maxLife);
          ctx.beginPath(); ctx.arc(d.x, d.y, d.size * 2.5, 0, Math.PI * 2); ctx.fillStyle = d.color + '15'; ctx.fill();
          ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2); ctx.fillStyle = d.color; ctx.fill();
          ctx.beginPath(); ctx.arc(d.x, d.y, d.size * 0.4, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.globalAlpha = alpha * 0.5; ctx.fill();
          ctx.globalAlpha = 1;
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
    <div className="canvas-frame paint w-full mx-auto" style={{ maxWidth: 800, aspectRatio: '8 / 5' }}>
      <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair"
        onPointerDown={e => setMouse(e, true)} onPointerMove={e => { if (mouseRef.current.active) setMouse(e, true); }}
        onPointerUp={e => setMouse(e, false)} onPointerLeave={() => { mouseRef.current.active = false; }} />
    </div>
  );
}
