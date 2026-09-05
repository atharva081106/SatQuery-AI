"use client";

import { useState, useEffect, useMemo } from "react";

export interface MissionImage {
  name: string;
  base64: string;
  label: string;
}

export interface SampleMission {
  id: string;
  title: string;
  category: "all" | "disaster" | "maritime" | "sar" | "urban";
  categoryLabel: string;
  tag: string;
  location: string;
  coordinates: string;
  sensors: string;
  resolution: string;
  pipeline: string;
  query: string;
  description: string;
  technicalNote: string;
  images: MissionImage[];
}

interface MissionPresetsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectMission: (mission: SampleMission, autoSubmit?: boolean) => void;
}

// Pre-packaged demo queries for instant zero-latency loading
const FALLBACK_MISSIONS: SampleMission[] = [
  {
    id: "uttarakhand_flood",
    title: "Uttarakhand Flash Flood Change Detection",
    category: "disaster",
    categoryLabel: "Disaster & Floods",
    tag: "BI-TEMPORAL CHANGE DETECTION",
    location: "Rishi Ganga Valley, Uttarakhand, India",
    coordinates: "30.4150° N, 79.7340° E",
    sensors: "Cartosat-2S / Sentinel-2 Bi-Temporal Pair",
    resolution: "0.65m / 10m Ground Resolution",
    pipeline: "Siamese UNet Change Head + Mask Overlays",
    query: "Run Change Detection between pre-flood baseline and post-flood event",
    description: "Catastrophic cloudburst and glacial lake outburst flood in Chamoli. Demonstrates bi-temporal difference mapping, swollen silt channel boundary delineation, and affected transport infrastructure isolation.",
    technicalNote: "Directly solves PS 26167 requirement: Bi-temporal satellite pair processing with pixel-level displaced terrain identification.",
    images: []
  },
  {
    id: "mumbai_port_recon",
    title: "Mumbai Harbor & Docks Strategic Recon",
    category: "maritime",
    categoryLabel: "Maritime Recon",
    tag: "HIGH-RES VQA & GROUNDING",
    location: "Jawaharlal Nehru Port, Navi Mumbai, India",
    coordinates: "18.9490° N, 72.9510° E",
    sensors: "Cartosat-2S Panchromatic + Multispectral",
    resolution: "0.65m Sub-Meter GSD",
    pipeline: "SatSegNet Grounding Head + Bounding Boxes",
    query: "Highlight industrial storage facilities, maritime docks, and cargo vessels",
    description: "Deep-water seaport terminal evaluation. Isolates commercial container ships, docking berths, and cylindrical petroleum liquid storage clusters with sub-meter spatial precision.",
    technicalNote: "Validates high-resolution panchromatic spatial grounding with zero false alarms across ocean-land boundaries.",
    images: []
  },
  {
    id: "bay_of_bengal_sar",
    title: "Bay of Bengal Monsoon Cloud Penetration",
    category: "sar",
    categoryLabel: "SAR Microwave",
    tag: "OPTICAL–SAR CROSS-MODAL FUSION",
    location: "Andaman Sea Maritime Corridor, India",
    coordinates: "12.3520° N, 92.7840° E",
    sensors: "Cartosat-3 Optical + RISAT-1 C-Band SAR",
    resolution: "5.4 GHz Microwave + 1.2m SAR Stripmap",
    pipeline: "Cross-Modal Dual-Encoder Attention",
    query: "Penetrate cloud cover using SAR radar backscatter channels and extract obscured maritime features",
    description: "Overcomes 100% thick monsoon cloud cover obscuring optical satellites by fusing synthetic aperture radar backscatter returns to pinpoint maritime vessels and island coastlines.",
    technicalNote: "Demonstrates ISRO RISAT-1 microwave radar capabilities for all-weather 24/7 disaster and strategic surveillance.",
    images: []
  },
  {
    id: "sambhar_salt_lake",
    title: "Sambhar Salt Lake Boundary Desiccation",
    category: "urban",
    categoryLabel: "Ecology & Wetland",
    tag: "WETLAND DELINEATION & RFC 7946",
    location: "Sambhar Lake, Rajasthan, India",
    coordinates: "26.9010° N, 75.0020° E",
    sensors: "Resourcesat-2 LISS-4 Multispectral",
    resolution: "5.8m GSD (Green, Red, NIR)",
    pipeline: "Multispectral Water Indices + Vector Contouring",
    query: "Detect water body boundary and calculate total wetland surface area in km²",
    description: "Ramsar wetland desiccation monitoring. Isolates hypersaline brine lagoons from industrial salt evaporation pans and computes accurate surface area vector polygons in RFC 7946 GeoJSON.",
    technicalNote: "Validates multispectral SWIR/NIR water indices and automatic polygon area calculation in square kilometers.",
    images: []
  },
  {
    id: "bengaluru_urban_sprawl",
    title: "Bengaluru Tech Corridor Urban Built-Up",
    category: "urban",
    categoryLabel: "Urban Expansion",
    tag: "LAND USE & DENSITY ESTIMATION",
    location: "Whitefield IT Corridor, Bengaluru, India",
    coordinates: "12.9698° N, 77.7499° E",
    sensors: "Cartosat-3 High-Resolution Panchromatic",
    resolution: "0.28m State-of-the-Art GSD",
    pipeline: "Multi-Class Semantic Segmentation (Built-Up vs Road)",
    query: "Detect built-up structures, commercial buildings, and calculate built-up density percentage",
    description: "Rapid urban densification analysis isolating tech park footprints, multi-lane arterial roads, and remaining vegetative buffers with built-up ratio percentage.",
    technicalNote: "Evaluates sub-30cm Cartosat-3 high-detail resolving power for smart city planning and tax boundary audits.",
    images: []
  }
];

export default function MissionPresetsModal({
  isOpen,
  onClose,
  onSelectMission
}: MissionPresetsModalProps) {
  const [missions, setMissions] = useState<SampleMission[]>(FALLBACK_MISSIONS);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string>("uttarakhand_flood");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [previewTileIndex, setPreviewTileIndex] = useState<number>(0);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Fetch full tile image payloads from backend
  useEffect(() => {
    if (!isOpen) return;

    const fetchMissions = async () => {
      setLoading(true);
      try {
        const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${backendUrl}/api/sample-missions`);
        if (res.ok) {
          const data = await res.json();
          if (data.missions && data.missions.length > 0) {
            const enhanced: SampleMission[] = data.missions.map((backendM: any) => {
              const fallback = FALLBACK_MISSIONS.find(f => f.id === backendM.id) || FALLBACK_MISSIONS[0];
              return {
                ...fallback,
                title: backendM.title || fallback.title,
                tag: backendM.tag || fallback.tag,
                location: backendM.location || fallback.location,
                sensors: backendM.sensors || fallback.sensors,
                query: backendM.query || fallback.query,
                description: backendM.description || fallback.description,
                images: backendM.images || []
              };
            });
            setMissions(enhanced);
          }
        }
      } catch (err) {
        console.warn("Using bundled demo queries", err);
      } finally {
        setLoading(false);
      }
    };

    fetchMissions();
  }, [isOpen]);

  const filteredMissions = useMemo(() => {
    if (activeCategory === "all") return missions;
    return missions.filter(m => m.category === activeCategory);
  }, [missions, activeCategory]);

  const currentMission = useMemo(() => {
    return missions.find(m => m.id === selectedId) || filteredMissions[0] || missions[0];
  }, [missions, selectedId, filteredMissions]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 animate-fade-in font-sans">
      <div className="bg-black border border-white/20 rounded-none w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        
        {/* SpaceX-Style Minimalist Header Bar */}
        <div className="px-6 py-4 border-b border-white/15 flex items-center justify-between bg-[#050505]">
          <div className="flex items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono tracking-widest text-white/50 uppercase">
                  ISRO / SAC — PS 26167
                </span>
                <span className="text-white/20">•</span>
                <span className="text-[10px] font-mono tracking-widest text-white/70 uppercase">
                  VERIFIED SCENARIOS
                </span>
              </div>
              <h2 className="text-sm sm:text-base font-bold text-white tracking-widest uppercase mt-0.5">
                DEMO QUERIES
              </h2>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 border border-white/20 flex items-center justify-center text-white/60 hover:text-white hover:border-white transition-colors cursor-pointer text-xs"
            title="Close (Esc)"
          >
            ✕
          </button>
        </div>

        {/* Minimal Category Filter Tabs */}
        <div className="px-6 py-2.5 bg-black border-b border-white/10 flex items-center gap-2 overflow-x-auto custom-scrollbar text-xs font-mono">
          <span className="text-white/40 uppercase text-[10px] tracking-widest mr-2 hidden sm:inline">FILTER:</span>
          {[
            { id: "all", label: "ALL" },
            { id: "disaster", label: "FLOODS & DISASTERS" },
            { id: "maritime", label: "MARITIME RECON" },
            { id: "sar", label: "SAR RADAR" },
            { id: "urban", label: "URBAN EXPANSION" }
          ].map(tab => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveCategory(tab.id)}
              className={`px-3 py-1 text-xs tracking-wider uppercase transition-all cursor-pointer whitespace-nowrap border ${
                activeCategory === tab.id
                  ? "bg-white text-black border-white font-bold"
                  : "bg-transparent text-white/60 border-white/15 hover:border-white/40 hover:text-white"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col md:flex-row divide-y md:divide-y-0 md:divide-x divide-white/10 min-h-0">
          
          {/* Left Column: Demo Queries List */}
          <div className="w-full md:w-5/12 p-4 flex flex-col gap-2 overflow-y-auto bg-black">
            <div className="flex items-center justify-between px-1 mb-1">
              <span className="text-[10px] font-mono tracking-widest text-white/40 uppercase">
                DEMO QUERIES ({filteredMissions.length})
              </span>
              {loading && (
                <span className="text-[10px] font-mono text-white/60 animate-pulse tracking-wider">
                  SYNCING...
                </span>
              )}
            </div>

            {filteredMissions.map(m => {
              const isSelected = m.id === currentMission?.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => {
                    setSelectedId(m.id);
                    setPreviewTileIndex(0);
                  }}
                  className={`text-left p-3.5 border transition-all cursor-pointer ${
                    isSelected
                      ? "bg-white/10 border-white text-white"
                      : "bg-transparent border-white/10 hover:border-white/30 text-white/80"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-[9px] font-mono tracking-widest text-white/60 uppercase">
                      {m.tag}
                    </span>
                    <span className="text-[9px] font-mono text-white/40">
                      {m.images.length > 0 ? `${m.images.length} TILES` : "READY"}
                    </span>
                  </div>

                  <div className="font-semibold text-xs sm:text-sm text-white uppercase tracking-wider">
                    {m.title}
                  </div>

                  <div className="text-[11px] text-white/50 truncate mt-1 font-mono">
                    {m.location}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right Column: Query Detail, Satellite Previews, and TOP LAUNCH BUTTON */}
          {currentMission && (
            <div className="w-full md:w-7/12 p-6 flex flex-col overflow-y-auto bg-[#040404]">
              
              {/* TOP ACTION BAR - LAUNCH ON TOP (SpaceX Style) */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pb-5 mb-5 border-b border-white/15">
                <button
                  type="button"
                  onClick={() => {
                    onSelectMission(currentMission, true);
                    onClose();
                  }}
                  className="flex-1 py-2.5 px-5 bg-white text-black hover:bg-white/90 text-xs font-mono font-bold tracking-widest uppercase transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg hover:scale-[1.01]"
                >
                  <span>▲</span>
                  <span>LAUNCH QUERY</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    onSelectMission(currentMission, false);
                    onClose();
                  }}
                  className="py-2.5 px-4 border border-white/30 hover:border-white text-white hover:bg-white/10 text-xs font-mono font-semibold tracking-widest uppercase transition-colors cursor-pointer text-center"
                >
                  LOAD INPUT ONLY
                </button>
              </div>

              <div className="flex flex-col gap-4">
                
                {/* Header & Description */}
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-[10px] font-mono text-white/70 uppercase tracking-widest border border-white/20 px-2 py-0.5">
                      {currentMission.tag}
                    </span>
                    <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">
                      {currentMission.pipeline}
                    </span>
                  </div>

                  <h3 className="text-base sm:text-lg font-bold text-white tracking-widest uppercase mt-1">
                    {currentMission.title}
                  </h3>

                  <p className="text-xs text-white/70 mt-1.5 leading-relaxed font-sans">
                    {currentMission.description}
                  </p>
                </div>

                {/* Tactical Parameters Grid (SpaceX Telemetry Style) */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs font-mono bg-white/[0.02] p-3 border border-white/10">
                  <div className="col-span-2 sm:col-span-1">
                    <span className="text-white/40 block text-[9px] uppercase tracking-widest">GEOLOCATION</span>
                    <span className="text-white font-medium text-[11px] truncate block">{currentMission.location}</span>
                    <span className="text-white/60 text-[10px]">{currentMission.coordinates}</span>
                  </div>
                  <div>
                    <span className="text-white/40 block text-[9px] uppercase tracking-widest">PAYLOAD SENSOR</span>
                    <span className="text-white font-medium text-[11px] truncate block">{currentMission.sensors}</span>
                  </div>
                  <div>
                    <span className="text-white/40 block text-[9px] uppercase tracking-widest">GROUND RESOLUTION</span>
                    <span className="text-white/90 font-medium text-[11px] truncate block">{currentMission.resolution}</span>
                  </div>
                </div>

                {/* Natural Language Mission Query Command */}
                <div className="border border-white/15 p-3 bg-black">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-mono text-white/60 uppercase tracking-widest">
                      MISSION QUERY PROMPT
                    </span>
                    <span className="text-[9px] font-mono text-white/40 uppercase">PS 26167 GROUNDING</span>
                  </div>
                  <p className="text-xs text-white font-mono leading-relaxed bg-white/[0.04] p-3 border border-white/10">
                    "{currentMission.query}"
                  </p>
                </div>

                {/* Satellite Tiles View */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-mono tracking-widest text-white/60 uppercase font-semibold">
                      CO-REGISTERED SATELLITE TILES ({currentMission.images.length || 2})
                    </span>
                  </div>

                  {currentMission.images.length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {currentMission.images.map((img, idx) => (
                        <div
                          key={idx}
                          onClick={() => setPreviewTileIndex(idx)}
                          className={`flex flex-col gap-1 cursor-pointer transition-all p-1 border ${
                            previewTileIndex === idx
                              ? "border-white bg-white/5"
                              : "border-white/15 bg-black hover:border-white/40"
                          }`}
                        >
                          <div className="aspect-video sm:aspect-square overflow-hidden border border-white/10 bg-black relative group">
                            <img
                              src={img.base64}
                              alt={img.label}
                              className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent opacity-80" />
                            <span className="absolute bottom-1.5 left-2 right-2 text-[9px] font-mono text-white/90 uppercase tracking-wider truncate">
                              {img.label}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="aspect-video border border-white/10 bg-black flex flex-col items-center justify-center p-6 text-center">
                      <span className="text-xs text-white/70 font-mono uppercase tracking-wider">
                        Synthetic payload calibrated for {currentMission.title}
                      </span>
                      <span className="text-[10px] text-white/40 font-mono mt-1 uppercase">
                        Click 'Launch Query' to execute analysis
                      </span>
                    </div>
                  )}
                </div>

                {/* SIH 2026 Evaluator Note */}
                <div className="text-[10px] font-mono text-white/60 bg-white/[0.02] p-2.5 border border-white/10">
                  <span className="text-white font-bold mr-1.5 tracking-wider uppercase">NOTE:</span>
                  <span>{currentMission.technicalNote}</span>
                </div>

              </div>

            </div>
          )}

        </div>

      </div>
    </div>
  );
}
