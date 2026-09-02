"use client";

import dynamic from 'next/dynamic';

const RealGlobe = dynamic(() => import('./RealGlobe'), { 
  ssr: false,
  loading: () => (
    <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ width: 800, height: 800, maxWidth: "100%", aspectRatio: 1 }} />
    </div>
  )
});

export default function CobeGlobe() {
  return <RealGlobe />;
}
