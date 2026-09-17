"use client";

import React, { useEffect, useState } from "react";
import { Download, X, WifiOff, Wifi, RefreshCw, Share, PlusSquare, Smartphone, Monitor, CheckCircle2 } from "lucide-react";

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
  const [isMobile, setIsMobile] = useState(false);
  const [isIOS, setIsIOS] = useState(false);
  const [showInstallBanner, setShowInstallBanner] = useState(false);
  const [showInstallModal, setShowInstallModal] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const [showOnlineToast, setShowOnlineToast] = useState(false);
  const [hasUpdate, setHasUpdate] = useState(false);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);

  useEffect(() => {
    // 1. Standalone / installed check
    const checkStandalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      (window.navigator as unknown as { standalone?: boolean }).standalone === true ||
      document.referrer.includes("android-app://");
    setIsStandalone(checkStandalone);

    // 2. Mobile and iOS detection
    const ua = navigator.userAgent;
    const isIOSDevice =
      /iPad|iPhone|iPod/.test(ua) &&
      !(window as unknown as { MSStream?: unknown }).MSStream;
    setIsIOS(isIOSDevice);

    const isMobileDevice =
      /Android|iPhone|iPad|iPod|Opera Mini|IEMobile|WPDesktop/i.test(ua) ||
      window.innerWidth < 768;
    setIsMobile(isMobileDevice);

    // 3. Register Service Worker
    if ("serviceWorker" in navigator && process.env.NODE_ENV !== "test") {
      navigator.serviceWorker
        .register("/sw.js")
        .then((reg) => {
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
      
      const dismissed = sessionStorage.getItem("pwa_banner_dismissed");
      if (!dismissed) {
        setShowInstallBanner(true);
      }
    };

    window.addEventListener("beforeinstallprompt", handleBeforeInstall);

    // 5. On Mobile: If not standalone and not dismissed, show mobile install badge
    const dismissed = sessionStorage.getItem("pwa_banner_dismissed");
    if (!checkStandalone && !dismissed) {
      // Show install banner on phone browser
      setShowInstallBanner(true);
    }

    // 6. Global custom trigger listener: window.dispatchEvent(new Event("open-pwa-install"))
    const handleOpenModal = () => {
      setShowInstallModal(true);
    };
    window.addEventListener("open-pwa-install", handleOpenModal);

    // 7. Network connectivity listeners
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
      window.removeEventListener("open-pwa-install", handleOpenModal);
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const handleInstallClick = async () => {
    if (installPrompt) {
      try {
        await installPrompt.prompt();
        const choice = await installPrompt.userChoice;
        if (choice.outcome === "accepted") {
          setShowInstallBanner(false);
          setShowInstallModal(false);
          setInstallPrompt(null);
        }
      } catch (err) {
        console.warn("Install prompt error:", err);
        setShowInstallModal(true);
      }
    } else {
      // Open step-by-step installation guide
      setShowInstallModal(true);
    }
  };

  const handleDismissBanner = () => {
    setShowInstallBanner(false);
    sessionStorage.setItem("pwa_banner_dismissed", "true");
  };

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

      {/* ── 4. Mobile & Desktop Install Floating Banner (Design System Compliant) ── */}
      {showInstallBanner && !isStandalone && (
        <aside
          aria-label="PWA Installation Prompt"
          className={`fixed z-[9998] animate-in fade-in duration-300 ${
            isMobile
              ? "bottom-4 left-4 right-4 safe-bottom"
              : "top-20 right-6 max-w-md w-full"
          }`}
        >
          <div className="flex items-center justify-between gap-3 p-3.5 sm:p-4 rounded-2xl bg-black/90 border border-white/20 text-white shadow-[0_10px_40px_rgba(0,0,0,0.9)] backdrop-blur-2xl font-mono">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-white/10 border border-white/20 flex items-center justify-center shrink-0">
                <Download className="w-5 h-5 text-white" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-bold tracking-wider text-xs uppercase text-white truncate">
                    SatQuery AI
                  </span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/15 text-cyan-300 font-mono tracking-widest uppercase border border-white/10 shrink-0">
                    APP
                  </span>
                </div>
                <div className="text-[10px] text-white/50 tracking-wider uppercase truncate mt-0.5">
                  Offline Maps &amp; Fullscreen
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleInstallClick}
                className="px-4 py-2 rounded-full bg-white text-black font-bold text-[11px] uppercase tracking-widest hover:bg-white/90 active:scale-95 transition-all cursor-pointer shadow-md shadow-white/10"
              >
                Install
              </button>
              <button
                type="button"
                onClick={handleDismissBanner}
                className="p-2 rounded-full border border-white/15 text-white/40 hover:text-white hover:border-white/30 transition-colors cursor-pointer"
                title="Dismiss"
                aria-label="Dismiss installation prompt"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </aside>
      )}

      {/* ── 5. Dedicated Step-by-Step Install Modal (iOS, Android, & Desktop) ── */}
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
                  Progressive Web Application (PWA)
                </div>
              </div>
            </div>

            {/* Platform Instructions */}
            {isIOS ? (
              // iOS Safari Instructions
              <div className="space-y-4 mb-6">
                <p className="text-xs text-white/70 leading-relaxed">
                  Install on your iPhone or iPad for borderless fullscreen mode, instant launch, and offline map caching:
                </p>

                <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-3.5 text-xs">
                  <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-cyan-400 shrink-0">
                    <Share className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">Step 1</span>
                    <span className="text-white font-medium">Tap the Share icon in the Safari navigation bar</span>
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
            ) : installPrompt ? (
              // Android / Desktop with Native Prompt
              <div className="space-y-4 mb-6">
                <p className="text-xs text-white/70 leading-relaxed">
                  Add SatQuery AI to your home screen or desktop application drawer for fast offline satellite inspection:
                </p>
                <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-2 text-xs">
                  <div className="flex items-center gap-2 text-white/80">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span>Borderless standalone window</span>
                  </div>
                  <div className="flex items-center gap-2 text-white/80">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span>Offline satellite map tile caching</span>
                  </div>
                  <div className="flex items-center gap-2 text-white/80">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span>Direct home screen quick shortcuts</span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleInstallClick}
                  className="w-full py-3 rounded-full bg-white text-black font-bold text-xs uppercase tracking-widest hover:bg-white/90 active:scale-95 transition-all cursor-pointer shadow-lg shadow-white/10"
                >
                  Confirm Installation
                </button>
              </div>
            ) : (
              // Browser Menu Instructions (Firefox, Samsung Internet, Desktop fallback)
              <div className="space-y-4 mb-6">
                <p className="text-xs text-white/70 leading-relaxed">
                  Install this site as an application using your browser menu:
                </p>
                <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-3.5 text-xs">
                  <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-white shrink-0 font-bold">
                    ⋮
                  </div>
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">Step 1</span>
                    <span className="text-white font-medium">Tap your browser menu (three dots top-right)</span>
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
