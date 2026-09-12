"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import GlobeCarousel3d from '@/components/GlobeCarousel3d';

const planetaryImages = [
  { src: "/demo/uttarakhand_pre_event_t1.png", title: "UTTARAKHAND PRE-FLOOD", desc: "Cartosat-2S Baseline" },
  { src: "/demo/uttarakhand_post_event_t2.png", title: "UTTARAKHAND POST-FLOOD", desc: "Change Detection Target" },
  { src: "/demo/mumbai_cartosat2s_optical.png", title: "MUMBAI HARBOR", desc: "High-Res Sub-Meter Optical" },
  { src: "/demo/cyclone_cloud_obscured_optical.png", title: "BAY OF BENGAL (OPTICAL)", desc: "100% Cloud Obscuration" },
  { src: "/demo/risat1_cband_radar_sar.png", title: "BAY OF BENGAL (SAR)", desc: "C-Band Cloud Penetrating Radar" },
  { src: "/demo/sambhar_lake_liss4.png", title: "SAMBHAR SALT LAKE", desc: "Resourcesat-2 LISS-4 Multispectral" },
  { src: "/demo/bengaluru_cartosat3.png", title: "BENGALURU URBAN", desc: "Cartosat-3 Urban Expansion" },
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
