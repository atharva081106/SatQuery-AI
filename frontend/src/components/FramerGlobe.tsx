"use client";

import dynamic from 'next/dynamic';

const RealGlobe = dynamic(() => import('./RealGlobe'), { 
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-[800px] h-[800px] max-w-full aspect-square rounded-full border border-cyan-500/20 bg-cyan-950/10 animate-pulse flex items-center justify-center">
        <span className="text-xs font-mono tracking-widest text-cyan-400/60 uppercase">INITIALIZING TELEMETRY...</span>
      </div>
    </div>
  )
});

export default function FramerGlobe() {
  return <RealGlobe />;
}
