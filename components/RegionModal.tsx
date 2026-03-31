'use client';

import { useEffect, useState } from 'react';
import type { RegionType } from '@/app/page';

const CFG = {
  wind:  { label: 'Wind Region',  instr: 'Click and drag to guide the swirling winds',   bg: 'linear-gradient(180deg,#4a8a4a,#2d5a2d)', preview: '/wind-preview.png' },
  fire:  { label: 'Fire Region',  instr: 'Click and drag to interact with the flames',    bg: 'linear-gradient(180deg,#e8772e,#b33d15)', preview: '/fire-preview.png' },
  water: { label: 'Water Region', instr: 'Click and drag to create ripples in the water', bg: 'linear-gradient(180deg,#2ca5d8,#0e6f96)', preview: '/water-preview.png' },
  paint: { label: 'Paint Region', instr: 'Click and drag to splash vibrant colors',       bg: 'linear-gradient(180deg,#a855f7,#7e22ce)', preview: '/paint-preview.png' },
};

interface Props {
  region: Exclude<RegionType, null>;
  onClose: () => void;
  onNavigate: (region: RegionType) => void;
  children: React.ReactNode;
}

export default function RegionModal({ region, onClose, onNavigate, children }: Props) {
  const [visible, setVisible] = useState(false);
  const c = CFG[region];

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const others = (['wind','fire','water','paint'] as const).filter(r => r !== region);

  return (
    <div className={`fixed inset-0 z-40 region-overlay transition-opacity duration-[350ms] ${visible ? 'opacity-100' : 'opacity-0'}`}>
      {/* Banner */}
      <div className="region-banner absolute top-[10px] left-1/2 -translate-x-1/2 z-[55]" style={{ background: c.bg }}>
        {c.label}
      </div>

      {/* CENTER the canvas using CSS grid — NO flex-1 stretching */}
      <div className={`absolute inset-0 grid place-items-center transition-all duration-500 ${
        visible ? 'scale-100 opacity-100' : 'scale-90 opacity-0'
      }`}>
        {children}
      </div>

      {/* Instruction */}
      <p className="absolute bottom-3.5 left-1/2 -translate-x-1/2 z-[55] text-[13px] text-white/[0.38] tracking-wide whitespace-nowrap pointer-events-none">
        {c.instr}
      </p>

      {/* Preview thumbnails — absolutely positioned, never affects canvas layout */}
      <div className="absolute right-3.5 top-1/2 -translate-y-1/2 flex flex-col gap-2.5 z-[55] hidden xl:flex">
        {others.map(r => (
          <button key={r} onClick={() => onNavigate(r)} className="preview-thumb group" title={CFG[r].label}>
            <img src={CFG[r].preview} alt={CFG[r].label} className="w-full h-full object-cover" />
            <span className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent py-0.5 font-silk text-[6px] text-white text-center tracking-wider">
              {r.toUpperCase()}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
