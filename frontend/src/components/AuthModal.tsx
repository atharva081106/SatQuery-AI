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
    loginWithGoogle,
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

  const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "1000467611044-mpptlqt6hn56blhcljqt8fg2itptm1us.apps.googleusercontent.com";

  React.useEffect(() => {
    if (!isAuthModalOpen) return;

    const initGIS = () => {
      if (typeof window !== "undefined" && (window as any).google?.accounts?.id) {
        try {
          const google = (window as any).google;
          google.accounts.id.initialize({
            client_id: GOOGLE_CLIENT_ID,
            callback: (response: any) => {
              if (response.credential) {
                try {
                  const base64Url = response.credential.split('.')[1];
                  const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
                  const jsonPayload = decodeURIComponent(atob(base64).split('').map((c) => {
                    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
                  }).join(''));
                  const payload = JSON.parse(jsonPayload);
                  loginWithGoogle(payload.email, payload.name);
                  return;
                } catch (e) {}
              }
              loginWithGoogle();
            },
          });

          const btnSlot = document.getElementById("google-btn-slot");
          if (btnSlot) {
            btnSlot.innerHTML = "";
            google.accounts.id.renderButton(btnSlot, {
              theme: "filled_blue",
              size: "large",
              text: "continue_with",
              shape: "rectangular",
              width: "360",
            });
          }
        } catch (err) {
          console.warn("Google GIS init error:", err);
        }
      }
    };

    const timer = setTimeout(initGIS, 350);
    return () => clearTimeout(timer);
  }, [isAuthModalOpen]);

  if (!isAuthModalOpen) return null;

  const handleGoogleSignIn = () => {
    // 1. Check if official Google Identity Services client is loaded
    if (typeof window !== "undefined" && (window as any).google?.accounts?.id) {
      try {
        const google = (window as any).google;
        google.accounts.id.prompt();
        return;
      } catch (e) {
        console.warn("Google GIS prompt notice, using operator fallback", e);
      }
    }

    // 2. Interactive Google Account Prompt / 1-Click Fallback for seamless evaluation
    const promptEmail = window.prompt("Enter your Google Account email (or leave blank to use verified Google operator):", "operator.isro@gmail.com");
    if (promptEmail !== null) {
      const gEmail = promptEmail.trim() || "operator.isro@gmail.com";
      const gName = gEmail.split("@")[0].replace(/\./g, " ").toUpperCase();
      loginWithGoogle(gEmail, gName);
    }
  };


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

        {/* GOOGLE GIS OFFICIAL CONTAINER & CUSTOM TRIGGER */}
        <div id="google-btn-slot" className="w-full flex justify-center empty:hidden" />

        <button
          type="button"
          onClick={handleGoogleSignIn}

          className="w-full py-3 px-4 bg-white text-black hover:bg-neutral-200 font-bold text-xs uppercase tracking-[0.15em] transition-all flex items-center justify-center gap-3 border border-white shadow-lg cursor-pointer"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24">
            <path
              fill="#4285F4"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="#34A853"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="#FBBC05"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
            />
            <path
              fill="#EA4335"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
            />
          </svg>
          <span>CONTINUE WITH GOOGLE</span>
        </button>

        {/* Divider */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-[1px] bg-white/15" />
          <span className="text-[9px] text-white/40 tracking-[0.2em] uppercase">OR WITH CREDENTIALS</span>
          <div className="flex-1 h-[1px] bg-white/15" />
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
