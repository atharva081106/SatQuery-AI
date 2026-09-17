"use client";

import React, { useEffect, useState } from "react";
import { Download, X, WifiOff, Wifi, RefreshCw, Share, PlusSquare } from "lucide-react";

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
  const [isStandalone, setIsStandalone] = useState(false);
  const [showInstallBanner, setShowInstallBanner] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const [showOnlineToast, setShowOnlineToast] = useState(false);
  const [hasUpdate, setHasUpdate] = useState(false);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);
  const [isIOS, setIsIOS] = useState(false);
  const [showIOSPrompt, setShowIOSPrompt] = useState(false);

  useEffect(() => {
    // 1. Standalone / installed check
    const checkStandalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      (window.navigator as unknown as { standalone?: boolean }).standalone === true;
    setIsStandalone(checkStandalone);

    // 2. iOS detection
    const isIOSDevice =
      /iPad|iPhone|iPod/.test(navigator.userAgent) &&
      !(window as unknown as { MSStream?: unknown }).MSStream;
    setIsIOS(isIOSDevice);

    // 3. Register Service Worker
    if ("serviceWorker" in navigator && process.env.NODE_ENV !== "test") {
      navigator.serviceWorker
        .register("/sw.js")
        .then((reg) => {
          console.log("[PWA] Service Worker registered with scope:", reg.scope);

          // Check if waiting worker exists
          if (reg.waiting) {
            setWaitingWorker(reg.waiting);
            setHasUpdate(true);
          }

          // Detect new updates
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

      // Reload when new SW takes control
      let refreshing = false;
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (!refreshing) {
          refreshing = true;
          window.location.reload();
        }
      });
    }

    // 4. Capture beforeinstallprompt (Chrome / Edge / Android)
    const handleBeforeInstall = (e: Event) => {
      e.preventDefault();
      const promptEvent = e as BeforeInstallPromptEvent;
      setInstallPrompt(promptEvent);
      // Don't show if user dismissed it in this session
      const dismissed = sessionStorage.getItem("pwa_install_dismissed");
      if (!dismissed) {
        setShowInstallBanner(true);
      }
    };

    window.addEventListener("beforeinstallprompt", handleBeforeInstall);

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
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!installPrompt) return;
    await installPrompt.prompt();
    const choice = await installPrompt.userChoice;
    if (choice.outcome === "accepted") {
      setShowInstallBanner(false);
      setInstallPrompt(null);
    }
  };

  const handleDismissBanner = () => {
    setShowInstallBanner(false);
    sessionStorage.setItem("pwa_install_dismissed", "true");
  };

  const handleUpdate = () => {
    if (waitingWorker) {
      waitingWorker.postMessage({ type: "SKIP_WAITING" });
    }
  };

  return (
    <>
      {/* ── 1. Offline Mode Floating Indicator ── */}
      {!isOnline && (
        <div className="fixed bottom-5 left-5 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-amber-500/20 border border-amber-500/40 text-amber-200 text-xs font-mono shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-bottom-3 duration-300">
          <WifiOff className="w-4 h-4 text-amber-400 shrink-0" />
          <span>Offline Mode // Cached shell &amp; tiles active</span>
        </div>
      )}

      {/* ── 2. Back Online Toast ── */}
      {isOnline && showOnlineToast && (
        <div className="fixed bottom-5 left-5 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 text-xs font-mono shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-bottom-3 duration-300">
          <Wifi className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>Connection Restored // Online</span>
        </div>
      )}

      {/* ── 3. App Update Notification ── */}
      {hasUpdate && (
        <div className="fixed bottom-5 right-5 z-50 max-w-sm flex items-center gap-3 p-4 rounded-2xl bg-[#090d16]/95 border border-sky-500/40 text-white shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-bottom-4 duration-300">
          <RefreshCw className="w-5 h-5 text-sky-400 shrink-0 animate-spin" />
          <div className="flex-1 text-xs">
            <div className="font-semibold text-white">App Update Available</div>
            <div className="text-white/60">Refresh to load the latest satellite models &amp; tools.</div>
          </div>
          <button
            onClick={handleUpdate}
            className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-black font-semibold text-xs transition-colors shrink-0 cursor-pointer"
          >
            Update
          </button>
        </div>
      )}

      {/* ── 4. Install App Floating Banner (Desktop & Android) ── */}
      {showInstallBanner && !isStandalone && (
        <div className="fixed top-20 right-5 z-50 max-w-md w-[calc(100vw-2.5rem)] sm:w-auto flex items-center justify-between gap-4 p-4 rounded-2xl bg-[#080d1a]/95 border border-white/15 text-white shadow-2xl backdrop-blur-2xl animate-in fade-in slide-in-from-top-4 duration-300">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-sky-500/15 border border-sky-400/30 flex items-center justify-center shrink-0">
              <Download className="w-5 h-5 text-sky-400" />
            </div>
            <div className="text-xs">
              <div className="font-semibold text-white flex items-center gap-1.5">
                Install SatQuery AI
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 font-mono">PWA</span>
              </div>
              <div className="text-white/60 text-[11px] line-clamp-1">
                Install as a standalone app with offline map caching.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleInstallClick}
              className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-black font-semibold text-xs transition-all shadow-md shadow-sky-500/20 cursor-pointer"
            >
              Install
            </button>
            <button
              onClick={handleDismissBanner}
              className="p-1.5 rounded-lg hover:bg-white/10 text-white/50 hover:text-white transition-colors cursor-pointer"
              title="Dismiss"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* ── 5. iOS Safari "Add to Home Screen" Helper Prompt ── */}
      {showIOSPrompt && isIOS && !isStandalone && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-[90vw] max-w-md p-4 rounded-2xl bg-[#080d1a]/95 border border-sky-500/30 text-white shadow-2xl backdrop-blur-2xl animate-in fade-in slide-in-from-bottom-4 duration-300">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-sky-400">Install SatQuery AI on iOS</span>
            <button onClick={() => setShowIOSPrompt(false)} className="text-white/50 hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-white/70 mb-3">
            Install this app on your iPhone or iPad for fullscreen mode and offline maps:
          </p>
          <div className="flex items-center gap-2 text-xs text-white/80 bg-white/5 p-2.5 rounded-xl border border-white/10">
            <span>Tap</span>
            <Share className="w-4 h-4 text-sky-400 inline" />
            <span>then select</span>
            <span className="font-semibold text-sky-300 flex items-center gap-1">
              Add to Home Screen <PlusSquare className="w-3.5 h-3.5 inline" />
            </span>
          </div>
        </div>
      )}
    </>
  );
}
