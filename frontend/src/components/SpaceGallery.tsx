"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import GlobeCarousel3d from '@/components/GlobeCarousel3d';

const planetaryImages = [
  { src: "/planets/mercury.jpg", title: "MERCURY", desc: "Inner Solar System" },
  { src: "/planets/venus.jpg", title: "VENUS", desc: "Atmospheric Phenomenon" },
  { src: "/planets/earth.jpg", title: "EARTH", desc: "The Blue Marble" },
  { src: "/planets/mars.jpg", title: "MARS", desc: "The Red Planet" },
  { src: "/planets/jupiter.jpg", title: "JUPITER", desc: "Gas Giant" },
  { src: "/planets/saturn.jpg", title: "SATURN", desc: "Ringed World" },
  { src: "/planets/uranus.jpg", title: "URANUS", desc: "Ice Giant" },
  { src: "/planets/neptune.jpg", title: "NEPTUNE", desc: "Outer Solar System" },
  { src: "/planets/pluto.jpg", title: "PLUTO", desc: "Kuiper Belt Object" },
  { src: "/planets/ceres.jpg", title: "CERES", desc: "Dwarf Planet" },
  { src: "/planets/moon.jpg", title: "THE MOON", desc: "Earth's Satellite" },
  { src: "/planets/io.jpg", title: "IO", desc: "Volcanic Moon of Jupiter" },
  { src: "/planets/europa.jpg", title: "EUROPA", desc: "Icy Moon of Jupiter" },
  { src: "/planets/ganymede.jpg", title: "GANYMEDE", desc: "Largest Moon in Solar System" },
  { src: "/planets/callisto.jpg", title: "CALLISTO", desc: "Heavily Cratered Moon" },
  { src: "/planets/titan.jpg", title: "TITAN", desc: "Saturn's Largest Moon" },
  { src: "/planets/enceladus.jpg", title: "ENCELADUS", desc: "Ice Geysers of Saturn" },
  { src: "/planets/triton.jpg", title: "TRITON", desc: "Retrograde Moon of Neptune" },
  { src: "/planets/sun.jpg", title: "THE SUN", desc: "G-Type Main-Sequence Star" },
  { src: "/planets/vesta.jpg", title: "VESTA", desc: "Protoplanet Asteroid" },
];

export default function SpaceGallery() {
  const [isMobile, setIsMobile] = React.useState(false);

  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 640);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return (
    <div className="w-full h-screen min-h-dvh h-[100dvh] bg-black relative z-10 overflow-hidden flex flex-col">
      {/* Background Globe Animation */}
      <div className="absolute inset-0 z-0 flex items-center justify-center opacity-30 pointer-events-none">
        <div className="w-[800px] sm:w-[1200px] h-[800px] sm:h-[1200px]">
          <FramerGlobe />
        </div>
      </div>
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-black via-transparent to-black pointer-events-none"></div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full h-full pt-12 sm:pt-16 px-4">
        <div className="w-[800px] h-[650px] sm:h-[800px] max-w-full max-h-[75vh]">
          <GlobeCarousel3d 
            images={planetaryImages} 
            background="transparent"
            radius={isMobile ? 220 : 350}
            tileWidth={isMobile ? 140 : 210}
            tileHeight={isMobile ? 95 : 140}
            count={25}
            speed={20}
            distance={isMobile ? 650 : 800}
            hideBack={true}
          />
        </div>
        
        <div className="text-center mt-4 sm:mt-8 safe-bottom">
          <p className="text-white/50 text-[10px] sm:text-xs tracking-widest uppercase font-mono">( Touch &amp; Drag or Scroll to Navigate )</p>
        </div>
      </div>
    </div>
  );
}
