"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface SwipeSliderProps {
  beforeImage: string;
  afterImage: string;
  beforeLabel?: string;
  afterLabel?: string;
}

export default function SwipeSlider({
  beforeImage,
  afterImage,
  beforeLabel = "BASELINE / BEFORE",
  afterLabel = "MODIFIED / AFTER"
}: SwipeSliderProps) {
  const [sliderPos, setSliderPos] = useState(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedX = Math.max(0, Math.min(x, rect.width));
    const percentage = (clampedX / rect.width) * 100;
    setSliderPos(percentage);
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  }, [isDragging, handleMove]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  }, [isDragging, handleMove]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("touchmove", handleTouchMove);
      window.addEventListener("touchend", handleMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

  return (
    <div className="flex flex-col gap-2 w-full select-none">
      {/* HEADER CONTROLS */}
      <div className="flex justify-between items-center text-[9px] font-mono tracking-widest text-white/50 px-1">
        <span className="flex items-center gap-1.5 text-white/80">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
          {beforeLabel}
        </span>
        <span className="text-white/40">DRAG CURTAIN TO COMPARE</span>
        <span className="flex items-center gap-1.5 text-white/80">
          {afterLabel}
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        </span>
      </div>

      {/* INTERACTIVE COMPARISON STAGE */}
      <div 
        ref={containerRef}
        onMouseDown={() => setIsDragging(true)}
        onTouchStart={() => setIsDragging(true)}
        className="relative w-full aspect-square max-h-[360px] rounded-xl overflow-hidden border border-[#3a3a3f] cursor-ew-resize bg-black/80 shadow-2xl"
      >
        {/* AFTER IMAGE (BACKGROUND LAYER) */}
        <img 
          src={afterImage} 
          alt={afterLabel} 
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />

        {/* BEFORE IMAGE (CLIPPED TOP LAYER) */}
        <div 
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
        >
          <img 
            src={beforeImage} 
            alt={beforeLabel} 
            className="absolute inset-0 w-full h-full object-cover"
          />
        </div>

        {/* DRAGGABLE DIVIDER LINE */}
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_12px_rgba(255,255,255,0.8)] pointer-events-none z-20 flex items-center justify-center"
          style={{ left: `${sliderPos}%` }}
        >
          {/* TACTICAL DRAG HANDLE */}
          <div className="w-7 h-7 rounded-full bg-black/90 border-2 border-white flex items-center justify-center shadow-2xl pointer-events-auto cursor-ew-resize text-white text-[10px] font-bold">
            <span>‹›</span>
          </div>
        </div>

        {/* BADGES ON CANVAS */}
        <div className="absolute top-3 left-3 pointer-events-none z-10">
          <span className="micro-cap text-[8px] bg-black/70 backdrop-blur-md px-2 py-1 rounded border border-white/20 text-white/90">
            {sliderPos > 20 ? beforeLabel : ""}
          </span>
        </div>
        <div className="absolute top-3 right-3 pointer-events-none z-10">
          <span className="micro-cap text-[8px] bg-black/70 backdrop-blur-md px-2 py-1 rounded border border-white/20 text-white/90">
            {sliderPos < 80 ? afterLabel : ""}
          </span>
        </div>
      </div>
    </div>
  );
}
