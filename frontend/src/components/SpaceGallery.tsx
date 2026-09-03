"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import MotionGallery from '@/components/MotionGallery';

const planetaryImages = [
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg",
    title: "EARTH / APOLLO",
    desc: "The Blue Marble, captured by Apollo 17."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg",
    title: "MARS / ROSETTA",
    desc: "True color image of Mars taken by the OSIRIS instrument."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/e/e2/Jupiter.jpg",
    title: "JUPITER / CASSINI",
    desc: "True color mosaic of Jupiter captured by Cassini."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/c/c7/Saturn_during_Equinox.jpg",
    title: "SATURN / EQUINOX",
    desc: "Saturn with its rings edge-on, captured by Cassini."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/e/e1/FullMoon2010.jpg",
    title: "LUNAR SURFACE",
    desc: "High resolution mosaic of the near side of the Moon."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/9/98/Andromeda_Galaxy_%28with_h-alpha%29.jpg",
    title: "ANDROMEDA GALAXY",
    desc: "Deep space observation of our neighboring spiral galaxy."
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
        <div className="w-full h-[55vh] max-h-[550px]">
          <MotionGallery 
            images={planetaryImages} 
            itemWidth={100} 
            scrollSpeed={0.8}
            scrollSmoothness={0.6}
            gap={32}
          />
        </div>
        
        <div className="text-center mt-8">
          <p className="text-white/50 text-xs tracking-widest uppercase font-mono">( Drag or Scroll to Navigate )</p>
        </div>
      </div>
    </div>
  );
}
