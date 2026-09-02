"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import BackgroundSlideshow from '@/components/BackgroundSlideshow';
import FadeInScroll from '@/components/FadeInScroll';
import FramerGlobe from '@/components/FramerGlobe';
import WebsiteLoader from '@/components/WebsiteLoader';
import SystemLoader from '@/components/SystemLoader';
export default function LandingPage() {
  const router = useRouter();
  const [showSplash, setShowSplash] = useState(true);
  const [flyOff, setFlyOff] = useState(false);
  const [enteringSystem, setEnteringSystem] = useState(false);

  const handleEnterSystem = (e: React.MouseEvent<HTMLAnchorElement>, path: string) => {
    e.preventDefault();
    setEnteringSystem(true);
    setTimeout(() => {
      router.push(path);
    }, 1400);
  };

  useEffect(() => {
    // Trigger spaceship flying away from left to right after 600ms
    const timer1 = setTimeout(() => {
      setFlyOff(true);
    }, 600);

    // Remove from DOM entirely after 2200ms (after ship has flown across from left to right)
    const timer2 = setTimeout(() => {
      setShowSplash(false);
    }, 2200);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return (
    <main className="bg-black text-white relative">
      
      {/* SPLASH SCREEN */}
      {showSplash && (
        <div className="fixed inset-0 z-[99999]">
          <WebsiteLoader isFlyingOff={flyOff} />
        </div>
      )}

      {/* SYSTEM ENTRY LOADER */}
      {enteringSystem && (
        <SystemLoader />
      )}

      {/* FIXED TOP NAV OVERLAY */}
      <nav className="fixed top-0 left-0 w-full flex justify-between items-center px-8 py-6 z-50 mix-blend-difference">
        <div className="display-lg tracking-widest text-white">
          SATQUERY AI.
        </div>
        <div className="flex gap-8 items-center">
          <Link href="/dashboard" className="micro-cap text-white hover:opacity-70 transition-opacity">
            BENCHMARKS
          </Link>
          <Link href="/gallery" className="micro-cap text-white hover:opacity-70 transition-opacity">
            GALLERY
          </Link>
          <Link href="/faq" className="micro-cap text-white hover:opacity-70 transition-opacity">
            FAQ
          </Link>
        </div>
      </nav>

      {/* BAND 1: HERO */}
      <section className="relative h-screen w-full flex items-center justify-center overflow-hidden">
        {/* Full Bleed Background */}
        <div className="absolute inset-0 z-0 bg-black">
          <div className="absolute inset-0 opacity-70">
            <FramerGlobe />
          </div>
          <div className="absolute inset-0 z-10 bg-gradient-to-t from-black via-transparent to-transparent opacity-80 pointer-events-none"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col items-center justify-end h-full pb-32 text-center">
          <FadeInScroll delay={200}>
            <h1 className="display-xxl mb-4">
              MAKING SENSE<br/>OF THE EARTH.
            </h1>
          </FadeInScroll>
          <FadeInScroll delay={400}>
            <p className="body-md uppercase tracking-[4px] opacity-70 mb-12">
              Multimodal Remote Sensing Image Analysis
            </p>
          </FadeInScroll>
          <FadeInScroll delay={600}>
            <a href="/query" onClick={(e) => handleEnterSystem(e, '/query')} className="button-ghost-on-dark w-48 text-center hover:bg-white hover:text-black cursor-pointer">
              ENTER SYSTEM
            </a>
          </FadeInScroll>
        </div>
        
        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-pulse opacity-50 z-10">
          <span className="micro-cap">SCROLL TO EXPLORE</span>
          <span className="text-xs">&darr;</span>
        </div>
      </section>

      {/* BAND 2: FEATURE - VQA */}
      <section className="relative h-screen w-full flex items-center justify-center overflow-hidden">
        {/* Full Bleed Background */}
        <div 
          className="absolute inset-0 z-0 bg-center bg-cover bg-no-repeat"
          style={{ backgroundImage: "url('https://upload.wikimedia.org/wikipedia/commons/2/2a/PSLV_C45_EMISAT_campaign_23.jpg')" }}
        >
          <div className="absolute inset-0 bg-black/40"></div>
          <div className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-black opacity-80"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col items-start justify-end h-full pb-32 px-8 md:px-24 w-full max-w-[1500px] mx-auto">
          <FadeInScroll>
            <div className="micro-cap mb-4 opacity-50">CAPABILITY 01</div>
            <h2 className="display-xl mb-6 max-w-3xl">
              NATURAL LANGUAGE<br/>QUERIES
            </h2>
            <p className="body-md opacity-80 max-w-xl">
              Interact with complex remote sensing data using everyday language. Our advanced agentic pipeline interprets your intent and extracts precise insights from vast geographical areas.
            </p>
          </FadeInScroll>
        </div>
      </section>

      {/* BAND 3: FEATURE - GROUNDING */}
      <section className="relative h-screen w-full flex items-center justify-center overflow-hidden">
        {/* Full Bleed Background */}
        <div 
          className="absolute inset-0 z-0 bg-center bg-cover bg-no-repeat"
          style={{ backgroundImage: "url('https://upload.wikimedia.org/wikipedia/commons/b/b4/GSLV_Mk_III_D2_on_Second_Launch_Pad_of_Satish_Dhawan_Space_Centre%2C_Sriharikota_%28SDSC_SHAR%29.jpg')" }}
        >
          <div className="absolute inset-0 bg-black/50"></div>
          <div className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-black opacity-80"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col items-start justify-end h-full pb-32 px-8 md:px-24 w-full max-w-[1500px] mx-auto">
          <FadeInScroll>
            <div className="micro-cap mb-4 opacity-50">CAPABILITY 02</div>
            <h2 className="display-xl mb-6 max-w-3xl">
              SPATIAL<br/>LOCALIZATION
            </h2>
            <p className="body-md opacity-80 max-w-xl mb-8">
              Identify and bound critical infrastructure, environmental changes, and specific geographical features directly onto the image canvas with millimeter precision.
            </p>
            <Link href="/query" className="button-ghost-on-dark hover:bg-white hover:text-black">
              TRY THE DEMO
            </Link>
          </FadeInScroll>
        </div>
        
        {/* OVERLAID MINIMAL FOOTER */}
        <div className="absolute bottom-8 w-full px-8 flex flex-col md:flex-row justify-center items-center gap-6 text-[10px] tracking-widest uppercase text-white/60 font-semibold z-20">
          <span>SATQUERY AI &copy; {new Date().getFullYear()}</span>
          <div className="flex flex-wrap justify-center items-center gap-6 md:gap-8">
            <a href="/query" onClick={(e) => handleEnterSystem(e, '/query')} className="hover:text-white transition-colors cursor-pointer">SYSTEM ACCESS</a>
            <Link href="/faq" className="hover:text-white transition-colors">FAQS</Link>
            <Link href="/dashboard" className="hover:text-white transition-colors">BENCHMARKS</Link>
            <Link href="#" className="hover:text-white transition-colors">DOCUMENTATION</Link>
            <Link href="#" className="hover:text-white transition-colors">PRIVACY POLICY</Link>
            <Link href="#" className="hover:text-white transition-colors">TERMS</Link>
          </div>
        </div>
      </section>
      
    </main>
  );
}
