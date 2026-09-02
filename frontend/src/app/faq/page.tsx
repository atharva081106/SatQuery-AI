"use client";

import Link from 'next/link';
import FadeInScroll from '@/components/FadeInScroll';
import FAQ from '@/components/FAQ';

export default function FAQPage() {
  return (
    <main className="bg-black text-white min-h-screen w-full relative overflow-y-auto custom-scrollbar flex flex-col">
      {/* FIXED TOP NAV */}
      <nav className="w-full flex justify-between items-center px-8 py-6 z-50 shrink-0">
        <Link href="/" className="display-lg tracking-widest text-white hover:opacity-70 transition-opacity">
          SATQUERY AI.
        </Link>
        <div className="flex gap-8 items-center">
          <Link href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 px-4 py-2 rounded-full flex items-center gap-2 cursor-pointer">
            <span>&larr;</span> BACK TO DASHBOARD
          </Link>
        </div>
      </nav>

      {/* MAIN KNOWLEDGE BASE CONTENT */}
      <div className="flex-1 w-full max-w-[1200px] mx-auto px-8 py-12 flex flex-col items-center">
        <FadeInScroll className="w-full text-left mb-12">
          <div className="micro-cap text-white/50 mb-2">01. KNOWLEDGE BASE</div>
          <h1 className="display-xl mb-4">
            FREQUENTLY ASKED<br />QUESTIONS.
          </h1>
          <p className="body-md opacity-70 max-w-2xl">
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
      <footer className="w-full px-8 py-8 border-t border-[#2a2a2f] flex flex-col md:flex-row justify-between items-center gap-4 text-[10px] tracking-widest uppercase text-white/50 shrink-0">
        <span>SATQUERY AI &copy; {new Date().getFullYear()}</span>
        <div className="flex flex-wrap justify-center items-center gap-6">
          <Link href="/dashboard" className="hover:text-white transition-colors">BENCHMARKS</Link>
          <Link href="/gallery" className="hover:text-white transition-colors">GALLERY</Link>
          <Link href="/query" className="hover:text-white transition-colors">SYSTEM ACCESS</Link>
          <Link href="#" className="hover:text-white transition-colors">DOCUMENTATION</Link>
        </div>
      </footer>
    </main>
  );
}
