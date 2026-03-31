'use client';

import { useState, useCallback } from 'react';
import AdventureMap from '@/components/AdventureMap';
import RegionModal from '@/components/RegionModal';
import SimulatorCanvas from '@/components/SimulatorCanvas';
import Sidebar from '@/components/Sidebar';

export type RegionType = 'wind' | 'fire' | 'water' | 'paint' | null;

export default function Home() {
  const [activeRegion, setActiveRegion] = useState<RegionType>(null);
  const [sparkles, setSparkles] = useState<{ id: number; x: number; y: number }[]>([]);

  const handleRegionClick = useCallback((region: RegionType, e?: React.MouseEvent) => {
    if (e) {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const newSparkles = Array.from({ length: 8 }, (_, i) => ({
        id: Date.now() + i,
        x: e.clientX - rect.left + (Math.random() - 0.5) * 70,
        y: e.clientY - rect.top + (Math.random() - 0.5) * 70,
      }));
      setSparkles(prev => [...prev, ...newSparkles]);
      setTimeout(() => setSparkles(prev => prev.filter(s => !newSparkles.find(ns => ns.id === s.id))), 700);
    }
    setTimeout(() => setActiveRegion(region), 220);
  }, []);

  const handleClose = useCallback(() => setActiveRegion(null), []);
  const handleNavigate = useCallback((region: RegionType) => setActiveRegion(region), []);

  return (
    <main className="relative w-full min-h-screen mossy-bg overflow-hidden">
      <AdventureMap onRegionClick={handleRegionClick} sparkles={sparkles} isHidden={activeRegion !== null} />

      {activeRegion && (
        <RegionModal region={activeRegion} onClose={handleClose} onNavigate={handleNavigate}>
          <Sidebar activeRegion={activeRegion} onNavigate={handleNavigate} onHome={handleClose} />
          <SimulatorCanvas key={activeRegion} region={activeRegion} />
        </RegionModal>
      )}

      <div className="fixed bottom-4 right-4 w-9 h-9 rounded-full bg-black/40 border-2 border-white/10 flex items-center justify-center cursor-pointer text-white/45 text-sm z-[100] hover:bg-black/60 hover:text-white transition-all font-silk font-bold">?</div>
    </main>
  );
}
