"use client";

import Link from 'next/link';
import FadeInScroll from '@/components/FadeInScroll';
import FAQ from '@/components/FAQ';

export default function FAQPage() {
  return (
    <main className="bg-black bg-cosmic-glow text-white min-h-screen min-h-[100dvh] w-full relative overflow-y-auto custom-scrollbar flex flex-col">
      {/* FIXED TOP NAV */}
      <nav className="w-full flex justify-between items-center pwa-safe-header z-50 shrink-0 border-b border-white/10">
        <Link href="/" className="text-base sm:text-xl md:display-lg tracking-widest text-white hover:opacity-70 transition-opacity font-mono font-bold">
          SATQUERY AI.
        </Link>
        <div className="flex gap-4 sm:gap-8 items-center">
          <Link href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full flex items-center gap-1.5 cursor-pointer text-[10px] sm:text-xs">
            <span>&larr;</span> <span className="hidden sm:inline">BACK TO DASHBOARD</span><span className="sm:hidden">HOME</span>
          </Link>
        </div>
      </nav>

      {/* MAIN KNOWLEDGE BASE CONTENT */}
      <div className="flex-1 w-full max-w-[1200px] mx-auto px-4 sm:px-8 py-6 sm:py-12 flex flex-col items-center">
        <FadeInScroll className="w-full text-left mb-6 sm:mb-12">
          <div className="micro-cap text-white/50 mb-2 text-[10px] sm:text-xs">01. KNOWLEDGE BASE</div>
          <h1 className="text-2xl sm:text-4xl md:display-xl mb-3 sm:mb-4">
            FREQUENTLY ASKED<br />QUESTIONS.
          </h1>
          <p className="body-md opacity-75 max-w-2xl text-xs sm:text-base leading-relaxed">
            Technical input scope, geospatial imagery format specifications, multimodal reasoning architecture, and ISRO/SAC operational compliance.
          </p>
        </FadeInScroll>

        <div className="w-full">
          <FadeInScroll delay={200}>
            <FAQ />
          </FadeInScroll>
        </div>
      </div>

      {/* MINIMAL FOOTER */}
      <footer className="w-full px-4 sm:px-8 py-6 sm:py-8 border-t border-[#2a2a2f] flex flex-col md:flex-row justify-between items-center gap-4 text-[9px] sm:text-[10px] tracking-widest uppercase text-white/50 shrink-0 pb-[calc(1.5rem+env(safe-area-inset-bottom))]">
        <span>SATQUERY AI &copy; {new Date().getFullYear()}</span>
        <div className="flex flex-wrap justify-center items-center gap-4 sm:gap-6">
          <Link href="/dashboard" className="hover:text-white transition-colors">BENCHMARKS</Link>
          <Link href="/gallery" className="hover:text-white transition-colors">GALLERY</Link>
          <Link href="/query" className="hover:text-white transition-colors">SYSTEM ACCESS</Link>
          <Link href="#" className="hover:text-white transition-colors">DOCUMENTATION</Link>
        </div>
      </footer>
    </main>
  );
}
