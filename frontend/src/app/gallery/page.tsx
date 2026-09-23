"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';

const GlobeCarousel3d = dynamic(() => import('@/components/GlobeCarousel3d'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-16 h-16 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
    </div>
  )
});

const RocketLaunchSequence = dynamic(() => import('@/components/RocketLaunchSequence'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-16 h-16 rounded-full border-2 border-orange-400 border-t-transparent animate-spin" />
    </div>
  )
});

const SATELLITE_GALLERY_IMAGES = [
  {
    src: "/gallery/photo-1446776811953-b23d57bd21aa.avif",
    alt: "Low Earth Orbit limb and solar array",
    title: "LOW EARTH ORBIT / LEO",
    desc: "Atmospheric limb and solar panel silhouette captured at 420km altitude."
  },
  {
    src: "/gallery/photo-1443456066412-3e3ea69ee37c.avif",
    alt: "Desert sand dunes and aeolian patterns",
    title: "DESERT TOPOLOGY",
    desc: "Multispectral optical capture of shifting aeolian ripples in the Rub' al Khali."
  },
  {
    src: "/gallery/photo-1517866184231-7ef94c2ea930.avif",
    alt: "Coastal shallows and coral bathymetry",
    title: "COASTAL BATHYMETRY",
    desc: "Visible spectrum depth mapping along turquoise barrier reef shallows."
  },
  {
    src: "/gallery/photo-1529788295308-1eace6f67388.avif",
    alt: "Polar glacial crevasses and ice sheets",
    title: "GLACIAL CREVASSE DYNAMICS",
    desc: "Cryosphere monitoring revealing thermal calving fissures in Greenland."
  },
  {
    src: "/gallery/photo-1534996858221-380b92700493.avif",
    alt: "Agricultural pivot crop rings",
    title: "PRECISION AGRICULTURE",
    desc: "Center-pivot irrigation fields analyzed for NDVI vegetation health."
  },
  {
    src: "/gallery/photo-1543722530-d2c3201371e7.avif",
    alt: "Active stratovolcano thermal plume",
    title: "VOLCANIC ASH SURVEILLANCE",
    desc: "Shortwave infrared thermal anomalies highlighting active basaltic extrusion."
  },
  {
    src: "/gallery/photo-1558158539-226f4a45f7b3.avif",
    alt: "Sediment outflow river delta",
    title: "ESTUARY HYDRAULICS",
    desc: "Suspended particulate matter dispersal into deep continental shelf."
  },
  {
    src: "/gallery/photo-1614729939124-032f0b56c9ce.avif",
    alt: "Deep cosmic horizon and starfield",
    title: "DEEP SPACE OPTICS",
    desc: "Sensor calibration view across cosmic stellar backdrop and galactic dust."
  },
  {
    src: "/gallery/photo-1670884307247-d1e905df653d.avif",
    alt: "Deep-space ground tracking antenna",
    title: "ISTRAC DEEP SPACE DISH",
    desc: "Ground segment 32-meter antenna maintaining telemetry lock with payloads."
  },
  {
    src: "/gallery/premium_photo-1677511580659-f5fa0675a547.avif",
    alt: "Radar SAR surface scattering",
    title: "SYNTHETIC APERTURE RADAR",
    desc: "C-band polarimetric microwave imaging piercing complete nocturnal cloud deck."
  },
  {
    src: "/gallery/premium_photo-1690571200236-0f9098fc6ca9.avif",
    alt: "Metropolitan urban growth footprint",
    title: "URBAN ARTERIAL MORPHOLOGY",
    desc: "Sub-meter multispectral pan-sharpened assessment of city growth vectors."
  },
  {
    src: "/gallery/premium_photo-1712039658659-7019cfe912e1.avif",
    alt: "Atmospheric aurora borealis ribbons",
    title: "IONOSPHERIC AURORA",
    desc: "Geomagnetic disturbance induced oxygen excitation ribbons seen from orbit."
  },
  {
    src: "/gallery/premium_photo-1722018576685-45a415a4ff67.avif",
    alt: "Tropical cyclonic storm vortex",
    title: "CYCLONE EYEWALL DYNAMICS",
    desc: "Geostationary infrared telemetry tracking Category-4 pressure gradient."
  }
];

export default function GalleryPage() {
  const [activeTab, setActiveTab] = useState<'sphere' | 'launch'>('sphere');
  const [selectedImage, setSelectedImage] = useState<typeof SATELLITE_GALLERY_IMAGES[0] | null>(null);

  return (
    <main className="bg-black bg-cosmic-glow text-white min-h-[100dvh] w-full relative overflow-x-hidden">
      {/* FIXED TOP NAV */}
      <nav className="fixed top-0 left-0 w-full flex justify-between items-center pwa-safe-header z-50 bg-black/40 backdrop-blur-md border-b border-white/10 px-4 sm:px-8 py-3">
        <Link href="/" className="text-sm sm:text-lg md:text-xl tracking-widest text-white hover:opacity-70 transition-opacity font-mono font-bold flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          SATQUERY AI // GALLERY
        </Link>

        {/* View Mode Toggle Pill */}
        <div className="flex items-center p-1 rounded-full bg-white/10 border border-white/20 backdrop-blur-lg">
          <button
            type="button"
            onClick={() => setActiveTab('sphere')}
            className={`px-3 sm:px-4 py-1.5 rounded-full text-[11px] sm:text-xs font-mono font-bold uppercase tracking-wider transition-all duration-300 cursor-pointer ${
              activeTab === 'sphere'
                ? 'bg-cyan-500 text-black shadow-lg shadow-cyan-500/25'
                : 'text-white/70 hover:text-white'
            }`}
          >
            3D Satellite Globe
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('launch')}
            className={`px-3 sm:px-4 py-1.5 rounded-full text-[11px] sm:text-xs font-mono font-bold uppercase tracking-wider transition-all duration-300 cursor-pointer ${
              activeTab === 'launch'
                ? 'bg-orange-500 text-white shadow-lg shadow-orange-500/25'
                : 'text-white/70 hover:text-white'
            }`}
          >
            Launch Sequence
          </button>
        </div>

        <div className="flex gap-4 sm:gap-6 items-center">
          <Link
            href="/"
            className="micro-cap text-white hover:bg-white hover:text-black transition-all border border-white/20 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full flex items-center gap-1.5 text-[10px] sm:text-xs font-mono uppercase"
          >
            <span>&larr;</span> <span className="hidden sm:inline">BACK TO HOME</span><span className="sm:hidden">HOME</span>
          </Link>
        </div>
      </nav>

      {/* VIEWPORT 1: 3D SATELLITE GLOBE CAROUSEL */}
      {activeTab === 'sphere' && (
        <section className="relative w-full h-[100dvh] flex flex-col items-center justify-center pt-16 select-none animate-in fade-in duration-300">
          {/* Header Info Overlay */}
          <div className="absolute top-20 sm:top-24 left-1/2 -translate-x-1/2 text-center pointer-events-none z-20 px-4 w-full max-w-xl">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-cyan-400 mb-1">
              OBSERVATION PAYLOAD SAMPLES
            </div>
            <h1 className="text-xl sm:text-3xl font-extrabold tracking-wider font-mono">
              3D ORBITAL IMAGERY SPHERE
            </h1>
            <p className="text-xs sm:text-sm text-white/60 font-mono mt-1">
              Drag to spin the sphere • Click any tile to inspect high-resolution telemetry
            </p>
          </div>

          {/* 3D Sphere Carousel Component */}
          <div className="w-full h-full max-w-[1200px] max-h-[85vh] flex items-center justify-center relative">
            <GlobeCarousel3d
              images={SATELLITE_GALLERY_IMAGES}
              background="transparent"
              radius={280}
              tileWidth={130}
              tileHeight={95}
              count={26}
              speed={16}
              distance={780}
              hideBack={true}
              cornerRadius={8}
            />
          </div>

          {/* Quick Imagery Thumbnail Strip */}
          <div className="absolute bottom-6 left-0 w-full px-4 flex flex-col items-center gap-2 z-20 pointer-events-auto">
            <div className="flex items-center gap-2 overflow-x-auto max-w-full p-2 bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl">
              {SATELLITE_GALLERY_IMAGES.map((img, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setSelectedImage(img)}
                  className="group relative w-12 h-9 rounded-lg overflow-hidden shrink-0 border border-white/20 hover:border-cyan-400 transition-all hover:scale-110 cursor-pointer"
                  title={img.title}
                >
                  <img src={img.src} alt={img.alt} className="w-full h-full object-cover" />
                  <span className="absolute inset-0 bg-black/30 group-hover:bg-transparent transition-colors" />
                </button>
              ))}
            </div>
            <span className="text-[10px] font-mono tracking-widest text-white/40 uppercase">
              13 TELEMETRY SAMPLES CAPTURED VIA MULTISPECTRAL &amp; SAR CONSTELLATIONS
            </span>
          </div>

          {/* Image Detail Modal */}
          {selectedImage && (
            <div
              className="fixed inset-0 z-[99999] bg-black/85 backdrop-blur-xl flex items-center justify-center p-4 animate-in fade-in duration-200"
              onClick={() => setSelectedImage(null)}
            >
              <div
                className="relative max-w-3xl w-full bg-neutral-900 border border-white/20 rounded-2xl overflow-hidden shadow-2xl flex flex-col"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="relative w-full h-[45vh] sm:h-[55vh] bg-black flex items-center justify-center">
                  <img
                    src={selectedImage.src}
                    alt={selectedImage.alt}
                    className="w-full h-full object-contain"
                  />
                  <button
                    type="button"
                    onClick={() => setSelectedImage(null)}
                    className="absolute top-4 right-4 w-9 h-9 rounded-full bg-black/70 border border-white/20 text-white flex items-center justify-center font-mono font-bold hover:bg-white hover:text-black transition-all cursor-pointer"
                  >
                    ✕
                  </button>
                </div>
                <div className="p-6 font-mono">
                  <div className="text-xs text-cyan-400 tracking-widest uppercase mb-1">
                    SATELLITE OBSERVATION METADATA
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold tracking-wider mb-2 text-white">
                    {selectedImage.title}
                  </h2>
                  <p className="text-xs sm:text-sm text-white/70 leading-relaxed mb-4">
                    {selectedImage.desc}
                  </p>
                  <div className="flex flex-wrap gap-2 pt-3 border-t border-white/10 text-[11px] text-white/60">
                    <span className="px-2.5 py-1 rounded bg-white/5 border border-white/10">🛰️ Low Earth Orbit Sensor</span>
                    <span className="px-2.5 py-1 rounded bg-white/5 border border-white/10">📊 16-Bit Radiometric Resolution</span>
                    <span className="px-2.5 py-1 rounded bg-white/5 border border-white/10">⚡ On-Device AI Calibrated</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      )}

      {/* VIEWPORT 2: 3D ROCKET LAUNCH SIMULATION */}
      {activeTab === 'launch' && (
        <section className="relative w-full h-[100dvh] animate-in fade-in duration-300">
          <RocketLaunchSequence />
        </section>
      )}
    </main>
  );
}
