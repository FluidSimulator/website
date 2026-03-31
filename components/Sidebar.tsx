'use client';

import { useState } from 'react';
import type { RegionType } from '@/app/page';

interface Props { activeRegion: RegionType; onNavigate: (r: RegionType) => void; onHome: () => void; }

const ICONS = [
  { id: 'home', svg: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg> },
  { id: 'water', svg: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg> },
  { id: 'fire', svg: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg> },
  { id: 'wind', svg: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg> },
  { id: 'paint', svg: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><circle cx="11" cy="11" r="2"/></svg> },
];

export default function Sidebar({ activeRegion, onNavigate, onHome }: Props) {
  const [shaking, setShaking] = useState<string | null>(null);

  const click = (id: string) => {
    setShaking(id); setTimeout(() => setShaking(null), 500);
    if (id === 'home') onHome();
    else onNavigate(id as RegionType);
  };

  return (
    <div className="absolute left-3.5 top-1/2 -translate-y-1/2 z-[55] flex flex-col gap-1.5 bg-black/[0.4] backdrop-blur-xl rounded-[14px] p-2 border border-white/5 animate-[sideL_0.45s_cubic-bezier(0.16,1,0.3,1)_both]"
      style={{ '--tw-translate-y': '-50%' } as any}>
      {ICONS.map(icon => (
        <button key={icon.id}
          className={`sidebar-icon ${icon.id === activeRegion ? 'active' : ''} ${shaking === icon.id ? 'animate-poke-shake' : ''}`}
          onClick={() => click(icon.id)}
          title={icon.id === 'home' ? 'Back to Map' : `${icon.id.charAt(0).toUpperCase() + icon.id.slice(1)} Region`}>
          {icon.svg}
        </button>
      ))}
    </div>
  );
}
