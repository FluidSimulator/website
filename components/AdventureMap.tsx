'use client';

import { useState } from 'react';
import type { RegionType } from '@/app/page';

interface Props {
  onRegionClick: (region: RegionType, e?: React.MouseEvent) => void;
  sparkles: { id: number; x: number; y: number }[];
  isHidden: boolean;
}

const REGIONS: { id: RegionType; label: string; x: string; y: string; w: string; h: string }[] = [
  { id: 'wind',  label: 'Wind Region',  x: '22%', y: '2%',  w: '32%', h: '32%' },
  { id: 'fire',  label: 'Fire Region',  x: '56%', y: '2%',  w: '40%', h: '48%' },
  { id: 'water', label: 'Water Region', x: '15%', y: '30%', w: '40%', h: '45%' },
  { id: 'paint', label: 'Paint Region', x: '56%', y: '50%', w: '40%', h: '45%' },
];

export default function AdventureMap({ onRegionClick, sparkles, isHidden }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <div className={`absolute inset-0 flex items-center justify-center z-[2] p-4 transition-all duration-[550ms] ease-[cubic-bezier(0.4,0,0.2,1)] ${
      isHidden ? 'opacity-0 scale-90 pointer-events-none' : 'opacity-100 scale-100'
    }`}>
      <div className="map-frame relative w-full max-w-[1000px] animate-[slideUp_0.8s_cubic-bezier(0.16,1,0.3,1)_both]">
        <img src="/map.png" alt="Adventure Map" draggable={false} className="w-full h-auto block select-none pointer-events-none" />

        {REGIONS.map(r => (
          <div
            key={r.id}
            className="map-region group"
            style={{ left: r.x, top: r.y, width: r.w, height: r.h }}
            onClick={e => onRegionClick(r.id, e)}
            onMouseEnter={() => setHovered(r.id as string)}
            onMouseLeave={() => setHovered(null)}
          >
            <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/[0.72] backdrop-blur-[10px] px-5 py-3 rounded-xl border-2 border-white/15 text-center whitespace-nowrap transition-all duration-[250ms] ease-[cubic-bezier(0.4,0,0.2,1)] pointer-events-none ${
              hovered === r.id ? 'opacity-100 scale-100' : 'opacity-0 scale-[0.85]'
            }`}>
              <h3 className="font-silk text-[11px] text-white tracking-wide">{r.label}</h3>
              <p className="text-[11px] text-white/40 mt-0.5">▸ Click to explore</p>
            </div>
          </div>
        ))}

        {/* Sparkle effects */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden z-20">
          {sparkles.map(s => (
            <div key={s.id} className="absolute w-3 h-3 bg-[#ffd700] animate-sparkle drop-shadow-[0_0_4px_rgba(255,215,0,0.6)]"
              style={{ left: s.x, top: s.y, clipPath: 'polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%)' }} />
          ))}
        </div>
      </div>
    </div>
  );
}
