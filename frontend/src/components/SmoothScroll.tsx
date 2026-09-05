"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { ReactLenis } from "lenis/react";
import "lenis/dist/lenis.css";

interface SmoothScrollProps {
  children: React.ReactNode;
}

export default function SmoothScroll({ children }: SmoothScrollProps) {
  const pathname = usePathname();

  // Only enable Lenis on landing page ('/').
  // Application pages (/query, /acquire, /dashboard, etc.) need 100% native
  // 2-finger trackpad, mouse wheel, and touch scroll without event interception.
  const isLandingPage = pathname === "/";

  if (!isLandingPage) {
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
