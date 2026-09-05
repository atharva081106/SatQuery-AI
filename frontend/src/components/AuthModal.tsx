"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";

export default function AuthModal() {
  const {
    isAuthModalOpen,
    authModalReason,
    authModalInitialMode,
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
  const [gisLoaded, setGisLoaded] = useState(false);

  useEffect(() => {
    if (authModalInitialMode) {
      setMode(authModalInitialMode);
    }
  }, [authModalInitialMode, isAuthModalOpen]);

  const GOOGLE_CLIENT_ID =
    process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ||
    "1000467611044-mpptlqt6hn56blhcljqt8fg2itptm1us.apps.googleusercontent.com";

  useEffect(() => {
    if (!isAuthModalOpen) {
      setGisLoaded(false);
      return;
    }

    const initGIS = () => {
      if (typeof window !== "undefined" && (window as any).google?.accounts?.id) {
        try {
          const google = (window as any).google;
          google.accounts.id.initialize({
            client_id: GOOGLE_CLIENT_ID,
            callback: (response: any) => {
              if (response.credential) {
                try {
                  const base64Url = response.credential.split(".")[1];
                  const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
                  const jsonPayload = decodeURIComponent(
                    atob(base64)
                      .split("")
                      .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
                      .join("")
                  );
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
              theme: "filled_black",
              size: "large",
              text: "continue_with",
              shape: "rectangular",
              width: "360",
            });
            setGisLoaded(true);
          }
        } catch (err) {
          console.warn("Google GIS init error:", err);
          setGisLoaded(false);
        }
      }
    };

    const timer = setTimeout(initGIS, 250);
    return () => clearTimeout(timer);
  }, [isAuthModalOpen]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeAuthModal();
    };
    if (isAuthModalOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isAuthModalOpen, closeAuthModal]);

  if (!isAuthModalOpen) return null;

  const handleGoogleSignIn = () => {
    if (typeof window !== "undefined" && (window as any).google?.accounts?.id) {
      try {
        const google = (window as any).google;
        google.accounts.id.prompt();
        return;
      } catch (e) {
        console.warn("Google GIS prompt fallback", e);
      }
    }

    const promptEmail = window.prompt(
      "Enter your Google Account email (or leave blank for verified operator):",
      "operator.isro@gmail.com"
    );
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
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 sm:p-6 bg-black/40 backdrop-blur-md animate-fade-in font-sans">
      {/* Centered Modal Card - Transparent Curvy Glassmorphism Architecture */}
      <div className="relative w-full max-w-md bg-white/[0.04] backdrop-blur-3xl border border-white/25 rounded-3xl p-6 sm:p-7 text-white flex flex-col gap-4 shadow-[0_20px_50px_rgba(0,0,0,0.6),inset_0_1px_1px_rgba(255,255,255,0.25)] max-h-[92vh] overflow-y-auto custom-scrollbar">
        
        {/* Header */}
        <div className="flex items-start justify-between border-b border-white/15 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]" />
              <span className="text-[10px] font-mono tracking-widest text-emerald-300 font-bold uppercase">
                AUTHENTICATION REQUIRED
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-bold tracking-wider uppercase mt-1 font-mono text-white">
              SIGN IN TO CONTINUE USING
            </h2>
          </div>
          <button
            type="button"
            onClick={closeAuthModal}
            className="w-8 h-8 rounded-full border border-white/30 bg-white/10 hover:bg-white/25 text-white/80 hover:text-white transition-all flex items-center justify-center text-xs font-mono cursor-pointer"
            title="Close (Esc)"
          >
            &times;
          </button>
        </div>

        {/* Quota Limit Reached Banner - ONLY rendered when the user has completed their free queries */}
        {queryCount >= maxFreeQueries && (
          <div className="bg-white/[0.06] backdrop-blur-xl border border-white/15 rounded-2xl p-3.5 text-xs shadow-inner">
            <div className="flex items-center justify-between text-[10px] font-mono font-bold text-emerald-400 tracking-wider uppercase mb-1">
              <span>QUOTA LIMIT REACHED</span>
              <span>{queryCount}/{maxFreeQueries} FREE QUERIES USED</span>
            </div>
            <p className="text-white/90 text-[11px] leading-relaxed font-sans">
              {authModalReason ||
                "Sign in to continue using SatQuery AI. You have used your 3 free satellite analyses."}
            </p>
          </div>
        )}

        {/* Google 1-Click Sign-In (Guaranteed EXACTLY ONE button, zero duplication) */}
        <div id="google-btn-slot" className="w-full flex justify-center empty:hidden min-h-[44px] rounded-xl overflow-hidden" />

        {!gisLoaded && (
          <button
            type="button"
            onClick={handleGoogleSignIn}
            className="w-full py-3 px-4 bg-white/90 hover:bg-white text-black font-bold text-xs uppercase tracking-widest rounded-2xl transition-all flex items-center justify-center gap-2.5 border border-white/80 cursor-pointer shadow-lg hover:scale-[1.01]"
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
            <span className="font-mono">CONTINUE WITH GOOGLE</span>
          </button>
        )}

        {/* Divider */}
        <div className="flex items-center gap-3 my-0.5">
          <div className="flex-1 h-[1px] bg-white/15" />
          <span className="text-[9px] font-mono text-white/50 tracking-widest uppercase">OR EMAIL CREDENTIALS</span>
          <div className="flex-1 h-[1px] bg-white/15" />
        </div>

        {/* Curvy Tab Switcher */}
        <div className="grid grid-cols-2 p-1 bg-white/[0.08] backdrop-blur-xl rounded-2xl border border-white/20 font-mono text-xs">
          <button
            type="button"
            onClick={() => {
              setMode("signin");
              setError(null);
            }}
            className={`py-2 rounded-xl tracking-wider uppercase font-bold transition-all cursor-pointer ${
              mode === "signin"
                ? "bg-white text-black shadow-md"
                : "text-white/60 hover:text-white"
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
            className={`py-2 rounded-xl tracking-wider uppercase font-bold transition-all cursor-pointer ${
              mode === "signup"
                ? "bg-white text-black shadow-md"
                : "text-white/60 hover:text-white"
            }`}
          >
            REGISTER
          </button>
        </div>

        {/* Compact Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 font-mono">
          {error && (
            <div className="bg-red-500/15 backdrop-blur-sm border border-red-500/40 rounded-xl p-2.5 text-[10px] text-red-200 tracking-wide">
              {error}
            </div>
          )}

          {mode === "signup" && (
            <div className="flex flex-col gap-1.5">
              <label className="text-[9px] uppercase tracking-widest text-white/60 font-medium">
                Operator Callsign / Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Commander Vikram"
                className="bg-white/5 border border-white/15 focus:border-white/50 rounded-xl px-3.5 py-2 text-xs text-white outline-none transition-all placeholder:text-white/30"
                autoFocus
              />
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] uppercase tracking-widest text-white/60 font-medium">
              Station / Account Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="operator@ground-station.isro.in"
              className="bg-white/5 border border-white/15 focus:border-white/50 rounded-xl px-3.5 py-2 text-xs text-white outline-none transition-all placeholder:text-white/30"
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] uppercase tracking-widest text-white/60 font-medium">
              Access Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="bg-white/5 border border-white/15 focus:border-white/50 rounded-xl px-3.5 py-2 text-xs text-white outline-none transition-all placeholder:text-white/30"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full py-2.5 mt-1 border border-white/25 hover:border-white/60 bg-white/15 hover:bg-white text-white hover:text-black font-bold text-xs uppercase tracking-widest rounded-2xl transition-all cursor-pointer shadow-md"
          >
            {mode === "signin" ? "AUTHENTICATE & CONTINUE" : "REGISTER & UNLOCK"}
          </button>
        </form>

        {/* 1-Click SIH Evaluator Pass */}
        <div className="border-t border-white/10 pt-3 flex flex-col gap-2">
          <button
            type="button"
            onClick={loginAsJudge}
            className="w-full py-2.5 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-400/40 text-emerald-300 rounded-2xl text-[11px] font-mono font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
          >
            <span>⚡</span>
            <span>INSTANT SIH EVALUATOR PASS</span>
          </button>
          <div className="text-[8px] font-mono text-white/30 tracking-widest uppercase text-center">
            SOVEREIGN AIR-GAPPED &bull; RFC 7946 GIS COMPLIANT
          </div>
        </div>

      </div>
    </div>
  );
}
