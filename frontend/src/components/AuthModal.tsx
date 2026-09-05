"use client";

import React, { useState } from "react";
import { useAuth } from "@/context/AuthContext";

export default function AuthModal() {
  const {
    isAuthModalOpen,
    authModalReason,
    closeAuthModal,
    login,
    signup,
    loginAsJudge,
    queryCount,
    mapCount,
    maxFreeQueries,
    maxFreeMapQueries,
  } = useAuth();

  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  if (!isAuthModalOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!email || !email.includes("@")) {
      setError("Please provide a valid operator email address.");
      return;
    }

    if (!password || password.length < 4) {
      setError("Password must contain at least 4 characters.");
      return;
    }

    if (mode === "signup") {
      if (!name.trim()) {
        setError("Please enter your name or callsign.");
        return;
      }
      signup(name.trim(), email.trim());
    } else {
      login(email.trim(), name.trim() || undefined);
    }
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-fadeIn">
      {/* Centered Modal Card */}
      <div className="relative w-full max-w-md bg-[#0a0a0c] border border-white/20 shadow-[0_0_50px_rgba(0,240,255,0.15)] p-6 sm:p-8 font-mono text-white flex flex-col gap-6">
        
        {/* Top Decorative Aerospace Border Accents */}
        <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-[#00F0FF]" />
        <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-[#00F0FF]" />
        <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-[#00F0FF]" />
        <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-[#00F0FF]" />

        {/* Header with Close */}
        <div className="flex items-start justify-between border-b border-white/10 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#00F0FF] animate-pulse" />
              <span className="text-[10px] tracking-[0.25em] text-[#00F0FF] uppercase font-bold">
                SECURITY CLEARANCE
              </span>
            </div>
            <h2 className="text-xl font-bold tracking-wider uppercase mt-1">
              {mode === "signin" ? "OPERATOR SIGN IN" : "REGISTER CALLSIGN"}
            </h2>
          </div>
          <button
            onClick={closeAuthModal}
            className="text-white/40 hover:text-white transition-colors p-1 text-sm uppercase tracking-widest"
            title="Close"
          >
            [ESC &times;]
          </button>
        </div>

        {/* Dynamic Quota Alert Banner */}
        <div className="bg-[#00F0FF]/10 border border-[#00F0FF]/30 p-3.5 text-xs">
          <div className="flex items-center justify-between text-[11px] font-bold text-[#00F0FF] tracking-widest uppercase mb-1">
            <span>GUEST QUOTA EXHAUSTED</span>
            <span>
              {queryCount >= maxFreeQueries ? `${queryCount}/${maxFreeQueries} QUERIES` : `${mapCount}/${maxFreeMapQueries} MAP SCANS`}
            </span>
          </div>
          <p className="text-white/80 text-[11px] leading-relaxed">
            {authModalReason ||
              "Sign in or register to unlock unlimited high-resolution satellite analyses, RFC 7946 GeoJSON exports, and ISRO Bhuvan overlays."}
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="grid grid-cols-2 border border-white/20 p-0.5 bg-white/5">
          <button
            type="button"
            onClick={() => {
              setMode("signin");
              setError(null);
            }}
            className={`py-2 text-xs tracking-[0.15em] uppercase font-bold transition-all ${
              mode === "signin"
                ? "bg-white text-black shadow-md"
                : "text-white/50 hover:text-white"
            }`}
          >
            SIGN IN
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("signup");
              setError(null);
            }}
            className={`py-2 text-xs tracking-[0.15em] uppercase font-bold transition-all ${
              mode === "signup"
                ? "bg-white text-black shadow-md"
                : "text-white/50 hover:text-white"
            }`}
          >
            CREATE ACCOUNT
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {error && (
            <div className="bg-red-500/20 border border-red-500/50 p-2 text-[11px] text-red-200 tracking-wider">
              {error}
            </div>
          )}

          {mode === "signup" && (
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] uppercase tracking-widest text-white/60">
                Operator Name / Designation
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Commander Vikram"
                className="bg-black/50 border border-white/20 px-3 py-2 text-xs text-white outline-none focus:border-[#00F0FF] transition-colors"
                autoFocus
              />
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] uppercase tracking-widest text-white/60">
              Station / Defense Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="operator@ground-station.isro.in"
              className="bg-black/50 border border-white/20 px-3 py-2 text-xs text-white outline-none focus:border-[#00F0FF] transition-colors"
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] uppercase tracking-widest text-white/60">
              Access Key / Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="bg-black/50 border border-white/20 px-3 py-2 text-xs text-white outline-none focus:border-[#00F0FF] transition-colors"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full py-3 mt-2 bg-white text-black hover:bg-[#00F0FF] hover:text-black font-bold text-xs uppercase tracking-[0.2em] transition-all shadow-[0_0_20px_rgba(255,255,255,0.2)]"
          >
            {mode === "signin" ? "AUTHENTICATE & UNLOCK" : "REGISTER & UNLOCK UNLIMITED"}
          </button>
        </form>

        {/* 1-Click Judge & Hackathon Demo Pass */}
        <div className="border-t border-white/10 pt-4 flex flex-col gap-2">
          <div className="text-[10px] text-white/40 tracking-widest uppercase text-center">
            SIH 2026 EVALUATOR ACCESS
          </div>
          <button
            type="button"
            onClick={loginAsJudge}
            className="w-full py-2.5 bg-emerald-500/10 border border-emerald-400/40 text-emerald-300 hover:bg-emerald-500/20 text-xs font-bold uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-2"
          >
            <span>⚡</span>
            <span>INSTANT JUDGE / EVALUATOR PASS</span>
          </button>
        </div>

        {/* Security watermark */}
        <div className="text-[9px] text-white/30 tracking-[0.2em] uppercase text-center">
          AIR-GAPPED SOVEREIGN TELEMETRY &bull; RFC 7946 GIS COMPLIANT
        </div>
      </div>
    </div>
  );
}
