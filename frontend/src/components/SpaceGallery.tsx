"use client";

import React from 'react';
import FramerGlobe from '@/components/FramerGlobe';
import GlobeCarousel3d from '@/components/GlobeCarousel3d';

const planetaryImages = [
  { src: "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?q=80&w=1500", title: "EARTH / LEO", desc: "The Earth from Low Earth Orbit." },
  { src: "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=1500", title: "DEEP SPACE", desc: "A stunning view of a distant nebula." },
  { src: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1500", title: "GLOBAL NETWORK", desc: "Digital representation of satellite communication." },
  { src: "https://images.unsplash.com/photo-1614729939124-03290b56c9ce?q=80&w=1500", title: "ANDROMEDA", desc: "Our neighboring spiral galaxy." },
  { src: "https://images.unsplash.com/photo-1614732414444-096e5f1122d5?q=80&w=1500", title: "STARS", desc: "The infinite expanse of the milky way." },
  { src: "https://images.unsplash.com/photo-1447433589675-4aaa56922e15?q=80&w=1500", title: "SPACE STATION", desc: "Orbital perspective." },
  { src: "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?q=80&w=1500", title: "MARS", desc: "The red planet." },
  { src: "https://images.unsplash.com/photo-1464802686167-b9394f569811?q=80&w=1500", title: "MILKY WAY", desc: "Galactic core." },
  { src: "https://images.unsplash.com/photo-1581822261290-991b38697739?q=80&w=1500", title: "LUNAR", desc: "The moon." },
  { src: "https://images.unsplash.com/photo-1610296669228-602fa0e269f5?q=80&w=1500", title: "BLACK HOLE", desc: "Event horizon." },
  { src: "https://images.unsplash.com/photo-1543722530-d2c31556620f?q=80&w=1500", title: "EARTHRISE", desc: "Earth from lunar orbit." },
  { src: "https://images.unsplash.com/photo-1518066000714-58c45f1a2c0a?q=80&w=1500", title: "CLUSTER", desc: "Galaxy cluster." },
  { src: "https://images.unsplash.com/photo-1506443697920-a616ddfcbb3a?q=80&w=1500", title: "SATELLITE", desc: "Orbital mechanics." },
  { src: "https://images.unsplash.com/photo-1454789476662-53eb23ba5907?q=80&w=1500", title: "METEOR", desc: "Shooting stars." },
  { src: "https://images.unsplash.com/photo-1481277542470-605612bd2d61?q=80&w=1500", title: "ASTRONAUT", desc: "Spacewalk." },
  { src: "https://images.unsplash.com/photo-1501862700950-18382cd41497?q=80&w=1500", title: "SOLAR", desc: "Solar flare." },
  { src: "https://images.unsplash.com/photo-1537429149323-b689a9f24e93?q=80&w=1500", title: "LAUNCH", desc: "Rocket launch." },
  { src: "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?q=80&w=1500", title: "AURORA", desc: "Northern lights." },
  { src: "https://images.unsplash.com/photo-1445905595283-214c6282111b?q=80&w=1500", title: "ASTEROID", desc: "Asteroid belt." },
  { src: "https://images.unsplash.com/photo-1513151233558-d860c5398176?q=80&w=1500", title: "ROVER", desc: "Lunar rover." },
  { src: "https://images.unsplash.com/photo-1457364887197-25e21fb5a0f6?q=80&w=1500", title: "JUPITER", desc: "Gas giant." },
  { src: "https://images.unsplash.com/photo-1614642261989-10521e15fa1d?q=80&w=1500", title: "SATURN", desc: "Ring system." },
  { src: "https://images.unsplash.com/photo-1582298538104-a6988bf2c1e8?q=80&w=1500", title: "EVA", desc: "Extravehicular activity." },
  { src: "https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?q=80&w=1500", title: "EXOPLANET", desc: "Distant world." },
  { src: "https://images.unsplash.com/photo-1446776858070-70c3d24ea263?q=80&w=1500", title: "DEEP FIELD", desc: "Billions of galaxies." },
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
