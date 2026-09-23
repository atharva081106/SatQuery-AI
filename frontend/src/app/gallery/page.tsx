"use client";

import React from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';

const RocketLaunchSequence = dynamic(() => import('@/components/RocketLaunchSequence'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-screen flex flex-col items-center justify-center bg-black font-mono">
      <div className="w-14 h-14 rounded-full border-2 border-white/70 border-t-transparent animate-spin mb-4" />
      <div className="text-xs tracking-widest text-white/70 uppercase animate-pulse">
        CALIBRATING LAUNCH TRAJECTORY ENGINE...
      </div>
    </div>
  )
});

export default function GalleryPage() {
  return (
    <main className="bg-black text-white min-h-[100dvh] w-full relative overflow-hidden select-none">
      {/* HUD NAV */}
      <nav className="fixed top-0 left-0 w-full flex justify-between items-center z-50 bg-black/40 backdrop-blur-md border-b border-white/10 px-4 sm:px-8 py-3 pointer-events-auto">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="text-sm sm:text-base md:text-lg tracking-widest text-white hover:text-white/70 transition-colors font-mono font-bold flex items-center gap-2"
          >
            <span className="w-2 h-2 rounded-full bg-white/80 animate-pulse" />
            SATQUERY AI // LAUNCH SEQUENCE
          </Link>
          <span className="hidden md:inline-block px-2.5 py-0.5 rounded text-[10px] font-mono tracking-widest uppercase bg-white/5 border border-white/20 text-white/70">
            PSLV-XL FLIGHT PROFILE
          </span>
        </div>

        <div className="flex gap-3 sm:gap-4 items-center">
          <Link
            href="/#features"
            className="hidden sm:inline-flex text-[11px] font-mono text-white/70 hover:text-white transition-colors uppercase tracking-wider"
          >
            SPECIFICATIONS
          </Link>
          <Link
            href="/"
            className="text-white hover:bg-white hover:text-black transition-all border border-white/20 px-3.5 py-1.5 rounded-full flex items-center gap-1.5 text-[11px] sm:text-xs font-mono uppercase tracking-wider"
          >
            <span>&larr;</span> <span>RETURN HOME</span>
          </Link>
        </div>
      </nav>

      {/* FULLSCREEN LAUNCH SEQUENCE */}
      <section className="relative w-full h-[100dvh]">
        <RocketLaunchSequence />
      </section>
    </main>
  );
}
