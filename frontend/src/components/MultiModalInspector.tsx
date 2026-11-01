"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { 
  Layers, 
  Sliders, 
  Columns, 
  Maximize2, 
  Minimize2, 
  Download, 
  ShieldCheck, 
  Zap, 
  Info,
  Sparkles
} from "lucide-react";

interface MultiModalInspectorProps {
  rawImage: string;
  maskImage: string;
  rawLabel?: string;
  maskLabel?: string;
  modelProvenance?: string;
  latencyMs?: number;
  confidence?: number;
  executionPath?: string;
  classes?: Array<{ name: string; color: string; percentage?: string | number }>;
}

export default function MultiModalInspector({
  rawImage,
  maskImage,
  rawLabel = "RAW SATELLITE SENSOR",
  maskLabel = "NEURAL PREDICTION MASK",
  modelProvenance,
  latencyMs,
  confidence,
  executionPath,
  classes
}: MultiModalInspectorProps) {
  const [activeTab, setActiveTab] = useState<"curtain" | "blend" | "side">("curtain");
  const [curtainPos, setCurtainPos] = useState<number>(50); // percentage 0-100
  const [blendOpacity, setBlendOpacity] = useState<number>(70); // percentage 0-100
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Curtain slider drag handlers
  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedX = Math.max(0, Math.min(x, rect.width));
    const percentage = (clampedX / rect.width) * 100;
    setCurtainPos(percentage);
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

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  return (
    <div className={`flex flex-col gap-3 w-full rounded-2xl bg-black/60 border border-white/15 p-3.5 backdrop-blur-xl shadow-2xl transition-all ${isFullscreen ? 'fixed inset-4 z-50 bg-black/95 max-w-none overflow-y-auto' : ''}`}>
      
      {/* TOP HUD BAR */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="text-[11px] font-mono font-bold tracking-wider uppercase text-white/90">
            Interactive Visual Inspector
          </span>
          {executionPath && (
            <span className="text-[9px] px-2 py-0.5 rounded-full bg-white/10 text-white/80 font-mono border border-white/15 uppercase font-semibold">
              {executionPath}
            </span>
          )}
        </div>

        {/* VIEW MODE SELECTOR TABS */}
        <div className="flex items-center bg-white/5 border border-white/15 rounded-xl p-1 gap-1">
          <button
            type="button"
            onClick={() => setActiveTab("curtain")}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-mono tracking-wider uppercase flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === "curtain"
                ? "bg-white text-black font-bold shadow-md"
                : "text-white/60 hover:text-white hover:bg-white/5"
            }`}
            title="Curtain Wipe / Swipe Comparison"
          >
            <Columns className="w-3 h-3" />
            <span>Curtain</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("blend")}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-mono tracking-wider uppercase flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === "blend"
                ? "bg-white text-black font-bold shadow-md"
                : "text-white/60 hover:text-white hover:bg-white/5"
            }`}
            title="Continuous Opacity Alpha Blend"
          >
            <Layers className="w-3 h-3" />
            <span>Blend ({blendOpacity}%)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("side")}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-mono tracking-wider uppercase flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === "side"
                ? "bg-white text-black font-bold shadow-md"
                : "text-white/60 hover:text-white hover:bg-white/5"
            }`}
            title="Side-by-Side Dual View"
          >
            <Sliders className="w-3 h-3" />
            <span>Dual</span>
          </button>

          <button
            type="button"
            onClick={toggleFullscreen}
            className="p-1 text-white/50 hover:text-white transition-colors cursor-pointer rounded-md hover:bg-white/10 ml-1"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Expanded View"}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* MODE 1: CURTAIN SLIDER */}
      {activeTab === "curtain" && (
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-[9px] font-mono tracking-wider text-white/50 px-1">
            <span className="flex items-center gap-1 text-white/80">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
              {rawLabel}
            </span>
            <span className="text-[8px] tracking-widest text-white/40 uppercase">Drag divider left/right</span>
            <span className="flex items-center gap-1 text-white/80">
              {maskLabel}
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            </span>
          </div>

          <div
            ref={containerRef}
            onMouseDown={() => setIsDragging(true)}
            onTouchStart={() => setIsDragging(true)}
            className="relative w-full aspect-square max-h-[460px] rounded-xl overflow-hidden border border-white/20 cursor-ew-resize bg-black/90 select-none shadow-inner"
          >
            {/* Background Layer: Mask Image */}
            <img
              src={maskImage}
              alt={maskLabel}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            />

            {/* Top Layer: Raw Satellite Image clipped by curtain */}
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ clipPath: `inset(0 ${100 - curtainPos}% 0 0)` }}
            >
              <img
                src={rawImage}
                alt={rawLabel}
                className="absolute inset-0 w-full h-full object-cover"
              />
            </div>

            {/* Draggable Divider */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_15px_rgba(255,255,255,0.9)] pointer-events-none z-20 flex items-center justify-center"
              style={{ left: `${curtainPos}%` }}
            >
              <div className="w-7 h-7 rounded-full bg-black/90 border-2 border-white flex items-center justify-center shadow-2xl pointer-events-auto cursor-ew-resize text-white text-[10px] font-bold">
                <span>‹›</span>
              </div>
            </div>

            {/* Labels in Corner */}
            <div className="absolute top-3 left-3 pointer-events-none z-10">
              <span className="text-[9px] font-mono bg-black/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/20 text-white/90 shadow-md">
                {curtainPos > 25 ? rawLabel : ""}
              </span>
            </div>
            <div className="absolute top-3 right-3 pointer-events-none z-10">
              <span className="text-[9px] font-mono bg-black/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/20 text-white/90 shadow-md">
                {curtainPos < 75 ? maskLabel : ""}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* MODE 2: ALPHA BLEND OPACITY SLIDER */}
      {activeTab === "blend" && (
        <div className="flex flex-col gap-2.5">
          {/* Opacity Control Header */}
          <div className="flex items-center justify-between gap-3 bg-white/5 border border-white/10 rounded-xl px-3 py-2">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-white/70 uppercase">Mask Alpha:</span>
              <span className="text-xs font-mono font-bold text-white bg-white/10 px-2 py-0.5 rounded border border-white/15">
                {blendOpacity}%
              </span>
            </div>

            <input
              type="range"
              min="0"
              max="100"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(Number(e.target.value))}
              className="flex-1 accent-white h-1.5 bg-white/20 rounded-lg cursor-pointer"
            />

            <div className="flex gap-1">
              {[0, 50, 100].map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setBlendOpacity(preset)}
                  className={`text-[9px] font-mono px-2 py-0.5 rounded cursor-pointer transition-colors ${
                    blendOpacity === preset
                      ? "bg-white text-black font-bold"
                      : "bg-white/5 text-white/60 hover:text-white"
                  }`}
                >
                  {preset}%
                </button>
              ))}
            </div>
          </div>

          {/* Layered Blend Canvas */}
          <div className="relative w-full aspect-square max-h-[460px] rounded-xl overflow-hidden border border-white/20 bg-black/90 shadow-inner">
            {/* Base Satellite Image */}
            <img
              src={rawImage}
              alt={rawLabel}
              className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Blended Neural Mask */}
            <img
              src={maskImage}
              alt={maskLabel}
              className="absolute inset-0 w-full h-full object-cover transition-opacity duration-75 mix-blend-screen"
              style={{ opacity: blendOpacity / 100 }}
            />

            {/* Corner Badge */}
            <div className="absolute top-3 left-3 pointer-events-none z-10 flex gap-2">
              <span className="text-[9px] font-mono bg-black/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/20 text-white/90">
                Overlay Alpha: {blendOpacity}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* MODE 3: SIDE-BY-SIDE DUAL VIEW */}
      {activeTab === "side" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
          <div className="flex flex-col gap-1.5">
            <span className="text-[10px] font-mono text-white/60 uppercase px-1">{rawLabel}</span>
            <div className="relative rounded-xl overflow-hidden border border-white/20 bg-black aspect-square">
              <img src={rawImage} alt={rawLabel} className="w-full h-full object-cover" />
            </div>
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-[10px] font-mono text-white/60 uppercase px-1">{maskLabel}</span>
            <div className="relative rounded-xl overflow-hidden border border-white/20 bg-black aspect-square">
              <img src={maskImage} alt={maskLabel} className="w-full h-full object-cover" />
            </div>
          </div>
        </div>
      )}

      {/* DYNAMIC CLASS LEGEND CHIPS */}
      {classes && classes.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-white/10">
          <span className="text-[9px] font-mono uppercase text-white/40 tracking-wider">Classification Legend:</span>
          {classes.map((cls, idx) => (
            <div
              key={idx}
              className="flex items-center gap-1.5 bg-white/5 border border-white/15 px-2.5 py-1 rounded-full text-[10px] font-mono text-white/90"
            >
              <span
                className="w-2 h-2 rounded-full shadow-sm"
                style={{ backgroundColor: cls.color }}
              />
              <span className="font-semibold">{cls.name}</span>
              {cls.percentage !== undefined && (
                <span className="text-white/50 text-[9px]">({cls.percentage})</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* BOTTOM TELEMETRY FOOTER */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-white/10 text-[10px] font-mono text-white/60">
        <div className="flex items-center gap-2">
          {modelProvenance && (
            <span className="text-white/80 font-medium">
              <span className="text-white/40">Model:</span> {modelProvenance}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {confidence !== undefined && (
            <span className="flex items-center gap-1 text-emerald-400">
              <ShieldCheck className="w-3 h-3" />
              <span>{(confidence * 100).toFixed(1)}% Conf</span>
            </span>
          )}
          {latencyMs !== undefined && (
            <span className="flex items-center gap-1 text-white/70">
              <Zap className="w-3 h-3 text-amber-400" />
              <span>{latencyMs.toFixed(1)}ms</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

