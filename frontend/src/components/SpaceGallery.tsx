"use client";

import React, { useState } from 'react';
import Image from 'next/image';

const planetaryImages = [
  {
    src: "https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?q=80&w=1000&auto=format&fit=crop",
    title: "EARTH / LEO",
    desc: "Low Earth Orbit view of cloud formations."
  },
  {
    src: "https://images.unsplash.com/photo-1614729939124-03290b56c9ce?q=80&w=800&auto=format&fit=crop",
    title: "MARS / ORBITER",
    desc: "Martian surface topology captured by orbital reconnaissance."
  },
  {
    src: "https://images.unsplash.com/photo-1614732414444-098e5e111a42?q=80&w=800&auto=format&fit=crop",
    title: "JUPITER / JUNO",
    desc: "Atmospheric storm systems and gas belts."
  },
  {
    src: "https://images.unsplash.com/photo-1614732484003-ef9881555dc3?q=80&w=800&auto=format&fit=crop",
    title: "SATURN / CASSINI",
    desc: "Ring system shadows cast across the northern hemisphere."
  },
  {
    src: "https://images.unsplash.com/photo-1522030299830-16b8d3d049fe?q=80&w=600&auto=format&fit=crop",
    title: "LUNAR SURFACE",
    desc: "Impact craters on the lunar far side."
  },
  {
    src: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1200&auto=format&fit=crop",
    title: "EARTH / NIGHT",
    desc: "Global illumination tracking urban infrastructure."
  },
  {
    src: "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?q=80&w=1000&auto=format&fit=crop",
    title: "ISS / SOLAR PANELS",
    desc: "Orbital infrastructure against the atmospheric limb."
  },
  {
    src: "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=900&auto=format&fit=crop",
    title: "DEEP SPACE",
    desc: "Stellar nursery and gas nebula imaging."
  },
  {
    src: "https://images.unsplash.com/photo-1475274047050-1d0c0975c63e?q=80&w=800&auto=format&fit=crop",
    title: "MILKY WAY",
    desc: "Galactic core observed from orbital telescope."
  },
  {
    src: "https://images.unsplash.com/photo-1541185933-ef5d8ed016c2?q=80&w=1000&auto=format&fit=crop",
    title: "MARS / ROVER",
    desc: "Ground-level structural terrain analysis."
  }
];

export default function SpaceGallery() {
  const [selectedImg, setSelectedImg] = useState<typeof planetaryImages[0] | null>(null);

  return (
    <div className="w-full min-h-screen bg-black pt-32 pb-16 px-8 relative z-10">
      
      {/* Title */}
      <div className="max-w-7xl mx-auto mb-12">
        <h1 className="text-4xl font-mono text-white tracking-[0.2em] font-bold mb-2 uppercase">Planetary Reconnaissance</h1>
        <p className="text-[#00F0FF] font-mono text-sm tracking-widest uppercase">Orbital & Deep Space Satellite Imagery Library</p>
      </div>

      {/* Masonry Grid */}
      <div className="max-w-7xl mx-auto columns-1 sm:columns-2 md:columns-3 lg:columns-4 gap-6 space-y-6">
        {planetaryImages.map((img, idx) => (
          <div 
            key={idx} 
            className="break-inside-avoid group relative cursor-pointer overflow-hidden border border-white/10 bg-black/40"
            onClick={() => setSelectedImg(img)}
          >
            <img 
              src={img.src} 
              alt={img.title} 
              className="w-full h-auto object-cover transform group-hover:scale-105 transition-transform duration-700 ease-in-out opacity-80 group-hover:opacity-100" 
              loading="lazy"
            />
            
            {/* Hover Overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4 pointer-events-none">
              <span className="text-[#00F0FF] font-mono text-xs tracking-widest mb-1">{img.title}</span>
              <span className="text-white/70 font-sans text-xs line-clamp-2">{img.desc}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox / Modal */}
      {selectedImg && (
        <div 
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90 backdrop-blur-md p-4 sm:p-8 animate-in fade-in duration-300"
          onClick={() => setSelectedImg(null)}
        >
          {/* Close Button */}
          <button 
            className="absolute top-8 right-8 text-white hover:text-[#00F0FF] font-mono tracking-widest uppercase text-sm border border-white/20 px-4 py-2 hover:border-[#00F0FF] transition-colors"
            onClick={(e) => { e.stopPropagation(); setSelectedImg(null); }}
          >
            Close [X]
          </button>

          {/* Modal Content */}
          <div 
            className="relative max-w-6xl w-full max-h-[85vh] flex flex-col items-center justify-center border border-white/10 bg-black shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <img 
              src={selectedImg.src} 
              alt={selectedImg.title} 
              className="w-full h-auto max-h-[75vh] object-contain"
            />
            <div className="w-full bg-black/80 border-t border-white/10 p-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
              <div>
                <h3 className="text-[#00F0FF] font-mono text-sm tracking-[0.2em]">{selectedImg.title}</h3>
                <p className="text-white/60 font-sans text-xs mt-1">{selectedImg.desc}</p>
              </div>
              <button 
                className="text-[10px] font-mono tracking-widest text-white/50 hover:text-white border border-white/20 px-3 py-1.5 transition-colors uppercase shrink-0"
                onClick={() => window.open(selectedImg.src, '_blank')}
              >
                View Full Resolution ↗
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
