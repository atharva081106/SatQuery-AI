"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import SystemLoader from "@/components/SystemLoader";

// Dynamically import MapExplorer to disable SSR, because Leaflet requires the window object
const MapExplorer = dynamic(() => import("@/components/MapExplorer"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-black flex items-center justify-center">
      <div className="micro-cap text-white animate-pulse">INITIALIZING MAPPING ENGINE...</div>
    </div>
  ),
});

export default function AcquirePage() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Simulate brief system boot sequence
    const timer = setTimeout(() => setIsLoaded(true), 800);
    return () => clearTimeout(timer);
  }, []);

  if (!isLoaded) {
    return <SystemLoader />;
  }

  return (
    <main className="h-screen w-screen bg-black relative overflow-hidden flex flex-col">
      
      {/* FIXED TOP NAV OVERLAY */}
      <nav className="absolute top-0 left-0 w-full flex justify-between items-center px-8 py-6 z-[1000] pointer-events-none">
        <div className="display-lg tracking-widest text-white pointer-events-auto mix-blend-difference drop-shadow-md">
          SATQUERY AI.
        </div>
        <div className="flex gap-4 items-center">
          <Link href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 bg-black/50 backdrop-blur-md px-4 py-2 flex items-center gap-2 pointer-events-auto">
            <span>&larr;</span> ABORT
          </Link>
        </div>
      </nav>

      {/* FULL BLEED MAP */}
      <div className="flex-1 relative z-0 w-full h-full">
        <MapExplorer />
      </div>
    </main>
  );
}
