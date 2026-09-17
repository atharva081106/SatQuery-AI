"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { WifiOff, RefreshCw, Globe, Compass, Database, ArrowRight, ShieldCheck } from "lucide-react";

export default function OfflinePage() {
  const [isOnline, setIsOnline] = useState(false);
  const [isChecking, setIsChecking] = useState(false);

  useEffect(() => {
    setIsOnline(navigator.onLine);

    const handleOnline = () => {
      setIsOnline(true);
      // Auto-reload after a brief moment when internet is restored
      setTimeout(() => {
        window.location.reload();
      }, 1200);
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const handleRetry = () => {
    setIsChecking(true);
    setTimeout(() => {
      setIsChecking(false);
      if (navigator.onLine) {
        window.location.reload();
      }
    }, 800);
  };

  return (
    <main className="bg-[#030712] text-white min-h-screen w-full relative overflow-y-auto flex flex-col items-center justify-between p-6 md:p-12">
      {/* Background radial glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-sky-500/10 rounded-full blur-[140px] pointer-events-none" />

      {/* Top Bar */}
      <header className="w-full max-w-4xl flex justify-between items-center z-10">
        <Link href="/" className="text-sm font-bold tracking-widest text-sky-400 hover:text-sky-300 transition-colors">
          SATQUERY AI // PWA
        </Link>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs">
          <span className={`w-2 h-2 rounded-full ${isOnline ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
          <span className="text-white/70 font-mono">
            {isOnline ? "CONNECTION RESTORED" : "STANDALONE OFFLINE MODE"}
          </span>
        </div>
      </header>

      {/* Center Card */}
      <section className="w-full max-w-xl my-auto py-12 flex flex-col items-center text-center z-10">
        {/* Radar Icon Illustration */}
        <div className="relative mb-8">
          <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-sky-500/20 to-indigo-500/20 border border-sky-400/30 flex items-center justify-center backdrop-blur-md">
            <WifiOff className="w-10 h-10 text-sky-400" />
          </div>
          <div className="absolute -inset-2 rounded-full border border-sky-400/20 animate-ping pointer-events-none" />
          <div className="absolute -inset-6 rounded-full border border-indigo-400/10 pointer-events-none" />
        </div>

        <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight mb-3">
          Satellite Uplink Suspended
        </h1>
        <p className="text-white/60 text-sm md:text-base max-w-md mb-8">
          You are currently in offline mode. Previously cached satellite map tiles, mission telemetry, and documentation remain available locally on your device.
        </p>

        {/* Retry Button */}
        <div className="flex flex-col sm:flex-row items-center gap-3 w-full justify-center mb-10">
          <button
            onClick={handleRetry}
            disabled={isChecking}
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-black font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg shadow-sky-500/20 cursor-pointer disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isChecking ? "animate-spin" : ""}`} />
            {isChecking ? "Pinging Network..." : "Check Connection & Retry"}
          </button>
          <Link
            href="/"
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-white/10 hover:bg-white/15 text-white font-medium text-sm flex items-center justify-center gap-2 border border-white/10 transition-colors"
          >
            <Globe className="w-4 h-4 text-sky-400" />
            Go to Cached Home
          </Link>
        </div>

        {/* Offline Features Container */}
        <div className="w-full rounded-2xl bg-white/[0.03] border border-white/10 p-5 text-left backdrop-blur-sm">
          <div className="text-xs uppercase tracking-wider text-white/40 font-mono mb-3 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Available Offline in this PWA
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Link
              href="/query"
              className="p-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 flex items-center justify-between group transition-colors"
            >
              <div className="flex items-center gap-3">
                <Compass className="w-4 h-4 text-sky-400" />
                <span className="text-xs font-medium">AI Query Presets</span>
              </div>
              <ArrowRight className="w-3.5 h-3.5 text-white/40 group-hover:text-white group-hover:translate-x-0.5 transition-all" />
            </Link>

            <Link
              href="/gallery"
              className="p-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 flex items-center justify-between group transition-colors"
            >
              <div className="flex items-center gap-3">
                <Database className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium">Cached Gallery</span>
              </div>
              <ArrowRight className="w-3.5 h-3.5 text-white/40 group-hover:text-white group-hover:translate-x-0.5 transition-all" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="w-full max-w-4xl text-center text-xs text-white/30 font-mono z-10 pt-6">
        SatQuery AI // ISRO SAC PS-26167 // Progressive Web Application v1.0
      </footer>
    </main>
  );
}
