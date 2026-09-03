"use client";

import Link from 'next/link';
import SpaceGallery from '@/components/SpaceGallery';

export default function GalleryPage() {
  return (
    <main className="bg-black text-white min-h-screen w-full relative">
      {/* FIXED TOP NAV */}
      <nav className="absolute top-0 left-0 w-full flex justify-between items-center px-8 py-6 z-50 mix-blend-difference pointer-events-none">
        <Link href="/" className="display-lg tracking-widest text-white hover:opacity-70 transition-opacity pointer-events-auto">
          SATQUERY AI.
        </Link>
        <div className="flex gap-8 items-center pointer-events-auto">
          <Link href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 px-4 py-2 rounded-full flex items-center gap-2">
            <span>&larr;</span> BACK TO DASHBOARD
          </Link>
        </div>
      </nav>

      <SpaceGallery />
    </main>
  );
}
