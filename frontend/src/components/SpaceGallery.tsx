"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import GlobeCarousel3d from '@/components/GlobeCarousel3d';

const planetaryImages = [
  {
    src: "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?q=80&w=1500",
    title: "EARTH / LEO",
    desc: "The Earth from Low Earth Orbit."
  },
  {
    src: "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=1500",
    title: "DEEP SPACE",
    desc: "A stunning view of a distant nebula."
  },
  {
    src: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1500",
    title: "GLOBAL NETWORK",
    desc: "Digital representation of satellite communication."
  },
  {
    src: "https://images.unsplash.com/photo-1614729939124-03290b56c9ce?q=80&w=1500",
    title: "ANDROMEDA",
    desc: "Our neighboring spiral galaxy."
  },
  {
    src: "https://images.unsplash.com/photo-1614732414444-096e5f1122d5?q=80&w=1500",
    title: "STARS",
    desc: "The infinite expanse of the milky way."
  }
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
