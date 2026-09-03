"use client";

import React, { useState } from 'react';
import FramerGlobe from '@/components/FramerGlobe';

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
    src: "https://upload.wikimedia.org/wikipedia/commons/e/e5/Venus-real_color.jpg",
    title: "VENUS / MARINER 10",
    desc: "Processed true color image of Venus."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/0/04/International_Space_Station_after_undocking_of_STS-132.jpg",
    title: "ISS / LEO",
    desc: "The International Space Station in Low Earth Orbit."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/9/98/Andromeda_Galaxy_%28with_h-alpha%29.jpg",
    title: "ANDROMEDA GALAXY",
    desc: "Deep space observation of our neighboring spiral galaxy."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg",
    title: "PILLARS OF CREATION",
    desc: "Eagle Nebula captured by the Hubble Space Telescope."
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/0/06/NGC_3372_a.jpg",
    title: "CARINA NEBULA",
    desc: "Massive star-forming region in the Carina–Sagittarius Arm."
  }
];

export default function SpaceGallery() {
  const [selectedImg, setSelectedImg] = useState<typeof planetaryImages[0] | null>(null);

  return (
    <div className="w-full min-h-screen bg-black pt-32 pb-16 px-8 relative z-10 overflow-hidden">
      
      {/* Background Globe Animation */}
      <div className="absolute inset-0 z-0 flex items-center justify-center opacity-30 pointer-events-none">
        <div className="w-[1200px] h-[1200px]">
          <FramerGlobe />
        </div>
      </div>
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-black via-transparent to-black pointer-events-none"></div>

      {/* Masonry Grid */}
      <div className="relative z-10 max-w-7xl mx-auto columns-1 sm:columns-2 md:columns-3 lg:columns-4 gap-6 space-y-6">
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
