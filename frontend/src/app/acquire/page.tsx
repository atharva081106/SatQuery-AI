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
    <main className="h-screen w-screen bg-white relative overflow-hidden flex flex-col">
      {/* FULL BLEED MAP */}
      <div className="flex-1 relative z-0 w-full h-full">
        <MapExplorer />
      </div>
    </main>
  );
}
