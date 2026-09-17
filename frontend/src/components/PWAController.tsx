"use client";

import React, { useEffect, useRef, useState } from "react";
import { Download, X, WifiOff, Wifi, RefreshCw, Share, PlusSquare, Smartphone } from "lucide-react";

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: "accepted" | "dismissed";
    platform: string;
  }>;
  prompt(): Promise<void>;
}

export default function PWAController() {
  const [installPrompt, setInstallPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const installPromptRef = useRef<BeforeInstallPromptEvent | null>(null);
  const [isIOS, setIsIOS] = useState(false);
  const [showInstallModal, setShowInstallModal] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const [showOnlineToast, setShowOnlineToast] = useState(false);
  const [hasUpdate, setHasUpdate] = useState(false);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);

  useEffect(() => {
    // 1. Mobile and iOS detection
    const ua = navigator.userAgent;
    const isIOSDevice =
      /iPad|iPhone|iPod/.test(ua) &&
      !(window as unknown as { MSStream?: unknown }).MSStream;
    setIsIOS(isIOSDevice);

    // 2. Register Service Worker
    if ("serviceWorker" in navigator && process.env.NODE_ENV !== "test") {
      navigator.serviceWorker
        .register("/sw.js")
        .then((reg) => {
          if (reg.waiting) {
            setWaitingWorker(reg.waiting);
            setHasUpdate(true);
          }

          reg.addEventListener("updatefound", () => {
            const newWorker = reg.installing;
            if (newWorker) {
              newWorker.addEventListener("statechange", () => {
                if (newWorker.state === "installed" && navigator.serviceWorker.controller) {
                  setWaitingWorker(newWorker);
                  setHasUpdate(true);
                }
              });
            }
          });
        })
        .catch((err) => {
          console.warn("[PWA] Service Worker registration failed:", err);
        });

      let refreshing = false;
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (!refreshing) {
          refreshing = true;
          window.location.reload();
        }
      });
    }

    // 3. Capture beforeinstallprompt (Chrome / Edge / Android)
    const handleBeforeInstall = (e: Event) => {
      e.preventDefault();
      const promptEvent = e as BeforeInstallPromptEvent;
      setInstallPrompt(promptEvent);
      installPromptRef.current = promptEvent;
    };

    window.addEventListener("beforeinstallprompt", handleBeforeInstall);

    // 4. 1-Click Direct Native Install Trigger Listener
    const handleTriggerInstall = async () => {
      if (installPromptRef.current) {
        try {
          // Immediately trigger native browser install sheet with zero intermediate popups
          await installPromptRef.current.prompt();
          const choice = await installPromptRef.current.userChoice;
          if (choice.outcome === "accepted") {
            setInstallPrompt(null);
            installPromptRef.current = null;
            setShowInstallModal(false);
          }
        } catch (err) {
          console.warn("Direct install prompt error:", err);
          setShowInstallModal(true);
        }
      } else {
        // Fallback for iOS Safari (which lacks beforeinstallprompt API) or unsupported browsers
        setShowInstallModal(true);
      }
    };
    window.addEventListener("open-pwa-install", handleTriggerInstall);

    // 5. Network connectivity listeners
    setIsOnline(navigator.onLine);

    const handleOnline = () => {
      setIsOnline(true);
      setShowOnlineToast(true);
      const timer = setTimeout(() => setShowOnlineToast(false), 3500);
      return () => clearTimeout(timer);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setShowOnlineToast(false);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("beforeinstallprompt", handleBeforeInstall);
      window.removeEventListener("open-pwa-install", handleTriggerInstall);
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const handleUpdate = () => {
    if (waitingWorker) {
      waitingWorker.postMessage({ type: "SKIP_WAITING" });
    }
  };

  return (
    <>
      {/* ── 1. Offline Mode Floating Indicator (OLED Minimalist) ── */}
      {!isOnline && (
        <div className="fixed bottom-5 left-5 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-black/90 border border-amber-500/50 text-amber-200 text-[11px] font-mono shadow-2xl backdrop-blur-2xl animate-in fade-in slide-in-from-bottom-3 duration-300">
          <WifiOff className="w-4 h-4 text-amber-400 shrink-0" />
          <span className="tracking-wider uppercase">OFFLINE MODE // CACHED ASSETS ACTIVE</span>
        </div>
      )}

      {/* ── 2. Back Online Toast ── */}
      {isOnline && showOnlineToast && (
        <div className="fixed bottom-5 left-5 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-black/90 border border-emerald-500/50 text-emerald-200 text-[11px] font-mono shadow-2xl backdrop-blur-2xl animate-in fade-in slide-in-from-bottom-3 duration-300">
          <Wifi className="w-4 h-4 text-emerald-400 shrink-0" />
          <span className="tracking-wider uppercase">TELEMETRY RESTORED // ONLINE</span>
        </div>
      )}

      {/* ── 3. App Update Notification (Aerospace Glassmorphism) ── */}
      {hasUpdate && (
        <div className="fixed bottom-5 right-5 z-50 max-w-sm flex items-center gap-3 p-4 rounded-2xl bg-black/95 border border-white/25 text-white shadow-2xl backdrop-blur-2xl animate-in fade-in slide-in-from-bottom-4 duration-300 font-mono">
          <RefreshCw className="w-5 h-5 text-cyan-400 shrink-0 animate-spin" />
          <div className="flex-1 text-xs">
            <div className="font-bold tracking-widest text-white uppercase text-[11px]">System Update Available</div>
            <div className="text-white/60 text-[10px] tracking-normal mt-0.5">Load the latest satellite models and tools.</div>
          </div>
          <button
            onClick={handleUpdate}
            className="px-4 py-2 rounded-full bg-white text-black font-bold text-xs uppercase tracking-widest hover:bg-white/90 active:scale-95 transition-all shrink-0 cursor-pointer shadow-md shadow-white/10"
          >
            Update
          </button>
        </div>
      )}

      {/* ── 4. Fallback iOS Safari / Unsupported Browser Instructions Modal ── */}
      {showInstallModal && (
        <div className="fixed inset-0 z-[99999] bg-black/85 backdrop-blur-2xl flex items-center justify-center p-4 animate-in fade-in duration-200 font-mono">
          <div className="relative w-full max-w-md rounded-3xl bg-[#070b14] border border-white/20 p-6 text-white shadow-2xl">
            {/* Close Button */}
            <button
              onClick={() => setShowInstallModal(false)}
              className="absolute top-5 right-5 p-2 rounded-full border border-white/15 text-white/50 hover:text-white hover:border-white transition-colors cursor-pointer"
              aria-label="Close modal"
            >
              <X className="w-4 h-4" />
            </button>

            {/* Header */}
            <div className="flex items-center gap-3.5 mb-5">
              <div className="w-12 h-12 rounded-2xl bg-white/10 border border-white/20 flex items-center justify-center shrink-0">
                <Smartphone className="w-6 h-6 text-white" />
              </div>
              <div>
                <div className="text-sm font-bold tracking-widest uppercase text-white">
                  Install SatQuery AI
                </div>
                <div className="text-[10px] text-white/50 tracking-wider uppercase mt-0.5">
                  Standalone Application
                </div>
              </div>
            </div>

            {/* Platform Instructions */}
            {isIOS ? (
              // iOS Safari 2-step visual guidance
              <div className="space-y-4 mb-6">
                <p className="text-xs text-white/70 leading-relaxed">
                  Install on your iPhone or iPad for fullscreen mode and offline map caching:
                </p>

                <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-3.5 text-xs">
                  <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-cyan-400 shrink-0">
                    <Share className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">Step 1</span>
                    <span className="text-white font-medium">Tap the Share icon in Safari&apos;s toolbar</span>
                  </div>
                </div>

                <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-3.5 text-xs">
                  <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-white shrink-0">
                    <PlusSquare className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">Step 2</span>
                    <span className="text-white font-medium">Scroll down and tap &quot;Add to Home Screen&quot;</span>
                  </div>
                </div>
              </div>
            ) : (
              // General browser fallback
              <div className="space-y-4 mb-6">
                <p className="text-xs text-white/70 leading-relaxed">
                  Install this site as an application directly from your browser menu:
                </p>
                <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-3.5 text-xs">
                  <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-white shrink-0 font-bold">
                    ⋮
                  </div>
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">Step 1</span>
                    <span className="text-white font-medium">Open your browser menu (top-right corner)</span>
                  </div>
                </div>
                <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-3.5 text-xs">
                  <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-cyan-400 shrink-0">
                    <Download className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">Step 2</span>
                    <span className="text-white font-medium">Select &quot;Install app&quot; or &quot;Add to Home screen&quot;</span>
                  </div>
                </div>
              </div>
            )}

            {/* Footer */}
            <div className="text-center pt-2 border-t border-white/10">
              <button
                type="button"
                onClick={() => setShowInstallModal(false)}
                className="text-[11px] text-white/50 hover:text-white uppercase tracking-widest cursor-pointer py-1"
              >
                Close [✕]
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
