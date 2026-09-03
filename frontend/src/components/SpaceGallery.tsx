"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import GlobeCarousel3d from '@/components/GlobeCarousel3d';

const planetaryImages = [
  { src: "/gallery/photo-1443456066412-3e3ea69ee37c.avif", title: "SPACE OBSERVATION", desc: "Local space imagery." },
  { src: "/gallery/photo-1446776811953-b23d57bd21aa.avif", title: "EARTH LEO", desc: "Local space imagery." },
  { src: "/gallery/photo-1517866184231-7ef94c2ea930.avif", title: "DEEP SPACE", desc: "Local space imagery." },
  { src: "/gallery/photo-1529788295308-1eace6f67388.avif", title: "STARS", desc: "Local space imagery." },
  { src: "/gallery/photo-1534996858221-380b92700493.avif", title: "GALAXY", desc: "Local space imagery." },
  { src: "/gallery/photo-1543722530-d2c3201371e7.avif", title: "NEBULA", desc: "Local space imagery." },
  { src: "/gallery/photo-1558158539-226f4a45f7b3.avif", title: "COSMOS", desc: "Local space imagery." },
  { src: "/gallery/photo-1614729939124-032f0b56c9ce.avif", title: "ASTRONOMY", desc: "Local space imagery." },
  { src: "/gallery/photo-1670884307247-d1e905df653d.avif", title: "CONSTELLATION", desc: "Local space imagery." },
  { src: "/gallery/premium_photo-1677511580659-f5fa0675a547.avif", title: "ORBITAL", desc: "Local premium space imagery." },
  { src: "/gallery/premium_photo-1690571200236-0f9098fc6ca9.avif", title: "SATELLITE", desc: "Local premium space imagery." },
  { src: "/gallery/premium_photo-1712039658659-7019cfe912e1.avif", title: "LUNAR", desc: "Local premium space imagery." },
  { src: "/gallery/premium_photo-1722018576685-45a415a4ff67.avif", title: "EXPLORATION", desc: "Local premium space imagery." },
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
