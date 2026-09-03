"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import GlobeCarousel3d from '@/components/GlobeCarousel3d';

const planetaryImages = [
  { src: "/planet.jpg", title: "PLANET", desc: "Local planetary surface." },
  { src: "/asteroid.jpg", title: "ASTEROID", desc: "Local asteroid belt." },
  { src: "/cyborg_frog.jpg", title: "CYBORG FROG", desc: "Local anomaly detected." }
];

export default function SpaceGallery() {
  return (
    <div className="w-full h-screen bg-black relative z-10 overflow-hidden flex flex-col">
      {/* Background Globe Animation */}
      <div className="absolute inset-0 z-0 flex items-center justify-center opacity-30 pointer-events-none">
        <div className="w-[1200px] h-[1200px]">
          <FramerGlobe />
        </div>
      </div>
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-black via-transparent to-black pointer-events-none"></div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full h-full pt-16">
        <div className="w-[800px] h-[800px] max-w-full max-h-[80vh]">
          <GlobeCarousel3d 
            images={planetaryImages} 
            background="transparent"
            radius={350}
            tileWidth={210}
            tileHeight={140}
            count={25}
            speed={20}
            distance={800}
            hideBack={true}
          />
        </div>
        
        <div className="text-center mt-8">
          <p className="text-white/50 text-xs tracking-widest uppercase font-mono">( Drag or Scroll to Navigate )</p>
        </div>
      </div>
    </div>
  );
}
