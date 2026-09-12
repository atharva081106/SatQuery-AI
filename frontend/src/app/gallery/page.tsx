"use client";

import Link from 'next/link';
import RocketLaunchSequence from '@/components/RocketLaunchSequence';

export default function GalleryPage() {
  return (
    <main className="bg-black text-white min-h-[100dvh] w-full relative">
      {/* FIXED TOP NAV */}
      <nav className="fixed top-0 left-0 w-full flex justify-between items-center px-4 sm:px-8 py-3 sm:py-6 z-50 mix-blend-difference pointer-events-none safe-top">
        <Link href="/" className="text-sm sm:text-xl md:display-lg tracking-widest text-white hover:opacity-70 transition-opacity pointer-events-auto font-mono font-bold">
          SATQUERY AI.
        </Link>
        <div className="flex gap-4 sm:gap-8 items-center pointer-events-auto">
          <Link href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full flex items-center gap-1.5 text-[10px] sm:text-xs">
            <span>&larr;</span> <span className="hidden sm:inline">BACK TO DASHBOARD</span><span className="sm:hidden">HOME</span>
          </Link>
        </div>
      </nav>

      {/* 3D Rocket Launch Sequence (Scroll Section) */}
      <RocketLaunchSequence />
    </main>
  );
}
