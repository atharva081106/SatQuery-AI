"use client";

import React, { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { ReactLenis } from "lenis/react";
import "lenis/dist/lenis.css";

interface SmoothScrollProps {
  children: React.ReactNode;
}

export default function SmoothScroll({ children }: SmoothScrollProps) {
  const pathname = usePathname();
  const [isTouchDevice, setIsTouchDevice] = useState(false);

  useEffect(() => {
    // Detect mobile touch screen to preserve native momentum scrolling
    if (typeof window !== "undefined") {
      const isTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0 || window.matchMedia("(pointer: coarse)").matches;
      setIsTouchDevice(isTouch);
    }
  }, []);

  // Only enable Lenis on desktop landing page ('/').
  // Mobile devices and application pages (/query, /acquire, /dashboard, etc.) need 100% native
  // touch scroll and momentum gestures.
  const isLandingPage = pathname === "/";

  if (!isLandingPage || isTouchDevice) {
    return <>{children}</>;
  }

  return (
    <ReactLenis
      root
      options={{
        lerp: 0.09,
        duration: 1.2,
        smoothWheel: true,
        wheelMultiplier: 1,
        touchMultiplier: 1.5,
        infinite: false,
        prevent: (node) => {
          return Boolean(
            node?.hasAttribute?.("data-lenis-prevent") ||
            node?.classList?.contains("custom-scrollbar") ||
            node?.classList?.contains("overflow-y-auto") ||
            node?.closest?.("[data-lenis-prevent]") ||
            node?.closest?.(".custom-scrollbar") ||
            node?.closest?.(".overflow-y-auto")
          );
        },
      }}
    >
      {children}
    </ReactLenis>
  );
}
