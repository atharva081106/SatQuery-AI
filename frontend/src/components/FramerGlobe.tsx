"use client";

import dynamic from 'next/dynamic';

const RealGlobe = dynamic(() => import('./RealGlobe'), { 
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-[850px] h-[850px] max-w-[90vw] max-h-[85vh] aspect-square rounded-full border border-blue-500/10 bg-blue-950/5 animate-pulse flex items-center justify-center" />
    </div>
  )
});

export default function FramerGlobe() {
  return <RealGlobe />;
}
