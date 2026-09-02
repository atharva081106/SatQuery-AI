"use client";

import { useEffect, useState, useRef } from "react";
import { usePathname } from "next/navigation";
import SystemLoader from "@/components/SystemLoader";

export default function PageTransitionEffect() {
  const pathname = usePathname();
  const [isTransitioning, setIsTransitioning] = useState(false);
  const prevPathname = useRef(pathname);

  useEffect(() => {
    // Only trigger if the pathname actually changes (ignoring initial mounts or query param changes)
    if (prevPathname.current === pathname) {
      return;
    }

    prevPathname.current = pathname;
    setIsTransitioning(true);
    
    const timer = setTimeout(() => {
      setIsTransitioning(false);
    }, 1400);

    return () => clearTimeout(timer);
  }, [pathname]);

  if (!isTransitioning) return null;

  return <SystemLoader />;
}
