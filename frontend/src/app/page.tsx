"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import BackgroundSlideshow from '@/components/BackgroundSlideshow';
import FadeInScroll from '@/components/FadeInScroll';
import FramerGlobe from '@/components/FramerGlobe';
import WebsiteLoader from '@/components/WebsiteLoader';
import SystemLoader from '@/components/SystemLoader';
import { useAuth } from '@/context/AuthContext';

export default function LandingPage() {
  const router = useRouter();
  const { isAuthenticated, user, openAuthModal, logout } = useAuth();
  const [showSplash, setShowSplash] = useState(true);
  const [flyOff, setFlyOff] = useState(false);
  const [enteringSystem, setEnteringSystem] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleEnterSystem = (e: React.MouseEvent<HTMLAnchorElement>, path: string) => {
    e.preventDefault();
    setMobileMenuOpen(false);
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

      {/* MOBILE NAVBAR (< 768px) */}
      <header className="fixed top-0 left-0 w-full flex justify-between items-center px-5 py-4 z-50 mix-blend-difference md:hidden">
        <div className="text-base tracking-[0.2em] font-mono font-bold text-white">
          SATQUERY AI.
        </div>
        <button
          type="button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="w-10 h-10 flex flex-col items-center justify-center gap-1.5 rounded-full border border-white/30 text-white cursor-pointer touch-manipulation"
          aria-label="Toggle mobile menu"
        >
          <span className={`w-5 h-0.5 bg-white transition-transform duration-300 ${mobileMenuOpen ? 'rotate-45 translate-y-2' : ''}`} />
          <span className={`w-5 h-0.5 bg-white transition-opacity duration-300 ${mobileMenuOpen ? 'opacity-0' : ''}`} />
          <span className={`w-5 h-0.5 bg-white transition-transform duration-300 ${mobileMenuOpen ? '-rotate-45 -translate-y-2' : ''}`} />
        </button>
      </header>

      {/* MOBILE FULLSCREEN MENU DRAWER */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-[9999] bg-black/95 backdrop-blur-2xl flex flex-col justify-between p-6 pt-20 safe-bottom animate-in fade-in duration-200 md:hidden font-mono">
          <div className="flex flex-col gap-6 text-left">
            <div className="text-[10px] tracking-widest uppercase text-white font-bold border-b border-white/15 pb-2">
              MISSION NAVIGATION
            </div>
            <Link
              href="/acquire"
              onClick={() => setMobileMenuOpen(false)}
              className="text-xl font-bold tracking-wider text-white hover:text-white/80 transition-colors py-1 flex items-center justify-between"
            >
              <span>SATELLITE MAP</span>
              <span className="text-xs text-white/40">&rarr;</span>
            </Link>
            <Link
              href="/query"
              onClick={(e) => handleEnterSystem(e, '/query')}
              className="text-xl font-bold tracking-wider text-white hover:text-white/80 transition-colors py-1 flex items-center justify-between"
            >
              <span>AI QUERY WORKSPACE</span>
              <span className="text-xs text-white/40">&rarr;</span>
            </Link>
            <Link
              href="/dashboard"
              onClick={() => setMobileMenuOpen(false)}
              className="text-xl font-bold tracking-wider text-white hover:text-white/80 transition-colors py-1 flex items-center justify-between"
            >
              <span>BENCHMARKS &amp; TELEMETRY</span>
              <span className="text-xs text-white/40">&rarr;</span>
            </Link>
            <Link
              href="/gallery"
              onClick={() => setMobileMenuOpen(false)}
              className="text-xl font-bold tracking-wider text-white hover:text-white/80 transition-colors py-1 flex items-center justify-between"
            >
              <span>3D SPACE GALLERY</span>
              <span className="text-xs text-white/40">&rarr;</span>
            </Link>
            <Link
              href="/faq"
              onClick={() => setMobileMenuOpen(false)}
              className="text-xl font-bold tracking-wider text-white hover:text-white/80 transition-colors py-1 flex items-center justify-between"
            >
              <span>SYSTEM FAQS</span>
              <span className="text-xs text-white/40">&rarr;</span>
            </Link>
            <button
              type="button"
              onClick={() => {
                setMobileMenuOpen(false);
                if (typeof window !== "undefined") {
                  window.dispatchEvent(new Event("open-pwa-install"));
                }
              }}
              className="text-xl font-bold tracking-wider text-white hover:text-white/80 transition-colors py-1 flex items-center justify-between cursor-pointer text-left"
            >
              <span>INSTALL APP</span>
              <span className="text-xs text-white/40">&rarr;</span>
            </button>
          </div>

          <div className="flex flex-col gap-3 pt-6 border-t border-white/15">
            {!isAuthenticated ? (
              <button
                type="button"
                onClick={() => {
                  setMobileMenuOpen(false);
                  openAuthModal("Sign in to access advanced earth observation intelligence.", "signin");
                }}
                className="w-full py-3 rounded-full border border-white bg-white text-black font-bold text-xs uppercase tracking-widest text-center"
              >
                SIGN IN / REGISTER
              </button>
            ) : (
              <div className="flex items-center justify-between bg-white/5 border border-white/15 p-3 rounded-xl">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                  <span className="text-xs text-white/80 font-bold uppercase">{user?.name || "OPERATOR"}</span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setMobileMenuOpen(false);
                    logout();
                  }}
                  className="text-xs text-white/60 hover:text-white uppercase tracking-wider underline"
                >
                  SIGN OUT
                </button>
              </div>
            )}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(false)}
              className="w-full py-2.5 text-center text-xs text-white/40 uppercase tracking-widest"
            >
              CLOSE MENU [✕]
            </button>
          </div>
        </div>
      )}

      {/* DESKTOP TOP NAV OVERLAY (>= 768px) */}
      <nav className="fixed top-0 left-0 w-full justify-between items-center px-8 py-6 z-50 mix-blend-difference hidden md:flex">
        <div className="display-lg tracking-widest text-white">
          SATQUERY AI.
        </div>
        <div className="flex gap-7 items-center">
          <Link href="/acquire" className="micro-cap text-white hover:opacity-70 transition-opacity font-semibold">
            SATELLITE MAP
          </Link>
          <Link href="/dashboard" className="micro-cap text-white hover:opacity-70 transition-opacity">
            BENCHMARKS
          </Link>
          <Link href="/gallery" className="micro-cap text-white hover:opacity-70 transition-opacity">
            GALLERY
          </Link>
          <button
            type="button"
            onClick={() => {
              if (typeof window !== "undefined") {
                window.dispatchEvent(new Event("open-pwa-install"));
              }
            }}
            className="micro-cap text-white hover:opacity-70 transition-opacity cursor-pointer font-semibold"
            title="Install SatQuery AI Standalone App"
          >
            INSTALL APP
          </button>

          {/* Small Sign In / Sign Up Button */}
          {!isAuthenticated ? (
            <button
              type="button"
              onClick={() => openAuthModal("Sign in to access advanced earth observation intelligence.", "signin")}
              className="micro-cap border border-white/30 hover:border-white text-white hover:bg-white hover:text-black px-3.5 py-1.5 rounded-full transition-all duration-200 cursor-pointer tracking-widest uppercase font-semibold text-[11px]"
              title="Sign in or register"
            >
              SIGN IN
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-[10px] px-2.5 py-1 rounded-full bg-white/20 text-white/80 border border-white/40 font-mono font-bold tracking-widest uppercase">
                {user?.name?.slice(0, 14).toUpperCase() || "OPERATOR"}
              </span>
              <button
                type="button"
                onClick={logout}
                className="text-[10px] text-white/50 hover:text-white uppercase tracking-wider underline cursor-pointer"
                title="Sign out"
              >
                Sign Out
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* BAND 1: HERO */}
      <section className="relative min-h-screen min-h-dvh h-[100dvh] w-full flex items-center justify-center overflow-hidden">
        {/* Full Bleed Background */}
        <div className="absolute inset-0 z-0 bg-black">
          <div className="absolute inset-0 opacity-70">
            <FramerGlobe />
          </div>
          <div className="absolute inset-0 z-10 bg-gradient-to-t from-black via-transparent to-transparent opacity-80 pointer-events-none"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col items-center justify-center sm:justify-end h-full pt-16 sm:pt-0 pb-16 sm:pb-32 px-4 text-center">

          <FadeInScroll delay={200}>
            <h1 className="display-xxl mb-3 sm:mb-4">
              MAKING SENSE<br/>OF THE EARTH.
            </h1>
          </FadeInScroll>
          <FadeInScroll delay={400}>
            <p className="body-md uppercase tracking-[2px] sm:tracking-[4px] opacity-75 mb-6 sm:mb-12 text-xs sm:text-base px-2 max-w-lg">
              Multimodal Remote Sensing Image Analysis
            </p>
          </FadeInScroll>
          <FadeInScroll delay={600}>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 w-full max-w-xs sm:max-w-none">
              <a href="/query" onClick={(e) => handleEnterSystem(e, '/query')} className="button-ghost-on-dark w-full sm:w-auto sm:min-w-[190px] text-center hover:bg-white hover:text-black cursor-pointer font-bold tracking-[2px] uppercase py-3 sm:py-2 text-xs sm:text-sm">
                TRY FOR FREE
              </a>
              <Link href="/acquire" className="w-full sm:w-auto px-6 py-3 sm:py-2 rounded-full border border-white/20 bg-white/5 hover:bg-white hover:text-black transition-all text-center text-xs font-mono font-bold tracking-widest uppercase text-white/80 backdrop-blur-sm">
                SATELLITE MAP &rarr;
              </Link>
            </div>
          </FadeInScroll>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-4 sm:bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 animate-pulse opacity-50 z-10 pointer-events-none">
          <span className="micro-cap text-[10px]">SCROLL TO EXPLORE</span>
          <span className="text-xs">&darr;</span>
        </div>
      </section>

      {/* BAND 2: FEATURE - VQA */}
      <section className="relative min-h-screen min-h-dvh h-[100dvh] w-full flex items-center justify-center overflow-hidden">
        {/* Full Bleed Background */}
        <div 
          className="absolute inset-0 z-0 bg-center bg-cover bg-no-repeat"
          style={{ backgroundImage: "url('https://upload.wikimedia.org/wikipedia/commons/2/2a/PSLV_C45_EMISAT_campaign_23.jpg')" }}
        >
          <div className="absolute inset-0 bg-black/40"></div>
          <div className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-black opacity-80"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col items-start justify-center sm:justify-end h-full py-16 sm:pb-32 px-5 sm:px-8 md:px-24 w-full max-w-[1500px] mx-auto">
          <FadeInScroll>
            <div className="p-5 sm:p-0 rounded-2xl sm:rounded-none bg-black/60 sm:bg-transparent backdrop-blur-md sm:backdrop-blur-none border border-white/15 sm:border-none shadow-2xl sm:shadow-none max-w-2xl w-full">
              <div className="micro-cap mb-2 sm:mb-3 opacity-60 text-[10px]">CAPABILITY 01</div>
              <h2 className="display-xl mb-3 sm:mb-6">
                NATURAL LANGUAGE<br/>QUERIES
              </h2>
              <p className="body-md opacity-80 text-xs sm:text-base leading-relaxed">
                Interact with complex remote sensing data using everyday language. Our advanced agentic pipeline interprets your intent and extracts precise insights from vast geographical areas.
              </p>
              <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-white/10 font-mono text-[10px]">
                <span className="px-2.5 py-1 rounded-md bg-white/10 text-white/90 font-semibold">⚡ &lt;10ms ONNX INT8</span>
                <span className="px-2.5 py-1 rounded-md bg-white/10 text-white/90 font-semibold">🎯 90.2% Accuracy</span>
                <span className="px-2.5 py-1 rounded-md bg-white/10 text-white/90 font-semibold">🛰️ Agentic VQA</span>
              </div>
            </div>
          </FadeInScroll>
        </div>
      </section>

      {/* BAND 3: FEATURE - GROUNDING */}
      <section className="relative min-h-screen min-h-dvh h-[100dvh] w-full flex items-center justify-center overflow-hidden">
        {/* Full Bleed Background */}
        <div 
          className="absolute inset-0 z-0 bg-center bg-cover bg-no-repeat"
          style={{ backgroundImage: "url('https://upload.wikimedia.org/wikipedia/commons/b/b4/GSLV_Mk_III_D2_on_Second_Launch_Pad_of_Satish_Dhawan_Space_Centre%2C_Sriharikota_%28SDSC_SHAR%29.jpg')" }}
        >
          <div className="absolute inset-0 bg-black/50"></div>
          <div className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-black opacity-80"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col items-start justify-center sm:justify-end h-full py-16 sm:pb-36 px-5 sm:px-8 md:px-24 w-full max-w-[1500px] mx-auto">
          <FadeInScroll>
            <div className="p-5 sm:p-0 rounded-2xl sm:rounded-none bg-black/60 sm:bg-transparent backdrop-blur-md sm:backdrop-blur-none border border-white/15 sm:border-none shadow-2xl sm:shadow-none max-w-2xl w-full">
              <div className="micro-cap mb-2 sm:mb-3 opacity-60 text-[10px]">CAPABILITY 02</div>
              <h2 className="display-xl mb-3 sm:mb-6">
                SPATIAL<br/>LOCALIZATION
              </h2>
              <p className="body-md opacity-80 mb-4 sm:mb-6 text-xs sm:text-base leading-relaxed">
                Identify and bound critical infrastructure, environmental changes, and specific geographical features directly onto the image canvas with millimeter precision.
              </p>
              <div className="flex flex-wrap gap-2 mb-4 pt-2 border-t border-white/10 font-mono text-[10px]">
                <span className="px-2.5 py-1 rounded-md bg-white/10 text-white/90 font-semibold">📐 Millimeter Precision</span>
                <span className="px-2.5 py-1 rounded-md bg-white/10 text-white/90 font-semibold">🛡️ Zero FPR Gate</span>
                <span className="px-2.5 py-1 rounded-md bg-white/10 text-white/90 font-semibold">🇮🇳 Bhuvan &amp; Sentinel</span>
              </div>
              <Link href="/query" className="button-ghost-on-dark inline-block hover:bg-white hover:text-black text-xs sm:text-sm py-2.5 px-6">
                TRY THE DEMO
              </Link>
            </div>
          </FadeInScroll>
        </div>
        
        {/* OVERLAID MINIMAL FOOTER */}
        <div className="absolute bottom-3 sm:bottom-8 w-full px-4 sm:px-8 flex flex-col md:flex-row justify-center items-center gap-2 sm:gap-6 text-[9px] sm:text-[10px] tracking-widest uppercase text-white/60 font-semibold z-20">
          <span>SATQUERY AI &copy; {new Date().getFullYear()}</span>
          <div className="flex flex-wrap justify-center items-center gap-4 md:gap-8">
            <a href="/query" onClick={(e) => handleEnterSystem(e, '/query')} className="hover:text-white transition-colors cursor-pointer">SYSTEM ACCESS</a>
            {!isAuthenticated ? (
              <button
                type="button"
                onClick={() => openAuthModal("Sign in to your SatQuery AI account.", "signin")}
                className="hover:text-white transition-colors cursor-pointer uppercase"
              >
                SIGN IN
              </button>
            ) : null}
            <Link href="/faq" className="hover:text-white transition-colors">FAQS</Link>
            <Link href="/dashboard" className="hover:text-white transition-colors">BENCHMARKS</Link>
            <button
              type="button"
              onClick={() => {
                if (typeof window !== "undefined") {
                  window.dispatchEvent(new Event("open-pwa-install"));
                }
              }}
              className="hover:text-white transition-colors cursor-pointer uppercase text-[9px] sm:text-[10px]"
            >
              INSTALL APP
            </button>
            <Link href="#" className="hover:text-white transition-colors">DOCUMENTATION</Link>
            <Link href="#" className="hover:text-white transition-colors">PRIVACY</Link>
            <Link href="#" className="hover:text-white transition-colors">TERMS</Link>
          </div>
        </div>
      </section>
      
    </main>
  );
}
