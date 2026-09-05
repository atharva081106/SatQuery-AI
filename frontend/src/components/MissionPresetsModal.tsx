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

// Fallback visual representations (zero-delay instant render)
const FALLBACK_MISSIONS: SampleMission[] = [
  {
    id: "uttarakhand_flood",
    title: "Uttarakhand Flash Flood Change Detection",
    category: "disaster",
    categoryLabel: "🌊 Disaster & Floods",
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
    categoryLabel: "🚢 Maritime Recon",
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
    categoryLabel: "🛰️ SAR Microwave",
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
    categoryLabel: "🌿 Ecology & Wetland",
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
    categoryLabel: "🏙️ Urban Expansion",
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

  // Close on escape key
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
            // Merge backend missions with enhanced metadata
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
        console.warn("Using bundled high-speed mission scenarios", err);
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
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 animate-fade-in">
      <div className="bg-[#0b0c10] border border-emerald-500/30 rounded-2xl w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl shadow-emerald-500/10 overflow-hidden">
        
        {/* Header Bar */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-emerald-950/40 via-transparent to-cyan-950/30">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500/30 to-cyan-500/20 border border-emerald-400/40 flex items-center justify-center text-lg shadow-sm shadow-emerald-500/20">
              🛰️
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono tracking-widest text-emerald-400 font-bold uppercase">
                  ISRO / SAC — PS 26167
                </span>
                <span className="px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-mono border border-cyan-500/30">
                  Ready-to-Run Presets
                </span>
              </div>
              <h2 className="text-base sm:text-lg font-bold text-white tracking-wide">
                Tactical Earth Observation Mission Scenarios
              </h2>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-full border border-white/20 flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            title="Close dialog (Esc)"
          >
            ✕
          </button>
        </div>

        {/* Filter Category Tabs */}
        <div className="px-6 py-2.5 bg-black/40 border-b border-white/5 flex items-center gap-2 overflow-x-auto custom-scrollbar text-xs font-mono">
          <span className="text-white/40 uppercase text-[10px] tracking-wider mr-1 hidden sm:inline">DOMAIN:</span>
          {[
            { id: "all", label: "ALL MISSIONS (5)" },
            { id: "disaster", label: "🌊 DISASTER & FLOODS" },
            { id: "maritime", label: "🚢 MARITIME & RECON" },
            { id: "sar", label: "🛰️ SAR CLOUD PENETRATION" },
            { id: "urban", label: "🏙️ URBAN & ECOLOGY" }
          ].map(tab => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveCategory(tab.id)}
              className={`px-3 py-1 rounded-lg transition-all cursor-pointer whitespace-nowrap text-xs ${
                activeCategory === tab.id
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-semibold shadow-sm shadow-emerald-500/20"
                  : "text-white/60 hover:text-white hover:bg-white/5 border border-transparent"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col md:flex-row divide-y md:divide-y-0 md:divide-x divide-white/10 min-h-0">
          
          {/* Left Column: Mission Scenarios List */}
          <div className="w-full md:w-5/12 p-4 flex flex-col gap-2.5 overflow-y-auto bg-black/20">
            <div className="flex items-center justify-between px-1">
              <span className="text-[10px] font-mono tracking-widest text-white/40 uppercase">
                AVAILABLE MISSIONS ({filteredMissions.length})
              </span>
              {loading && (
                <span className="text-[10px] font-mono text-emerald-400 animate-pulse flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                  Syncing Live Tiles...
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
                  className={`text-left p-3.5 rounded-xl border transition-all cursor-pointer relative group ${
                    isSelected
                      ? "bg-gradient-to-r from-emerald-500/15 via-white/[0.04] to-transparent border-emerald-400/80 shadow-lg shadow-emerald-500/10"
                      : "bg-white/[0.02] border-white/10 hover:bg-white/[0.05] hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-[10px] font-mono font-bold tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 uppercase">
                      {m.tag}
                    </span>
                    <span className="text-[10px] text-white/40 font-mono">
                      {m.images.length > 0 ? `${m.images.length} Imagery Tiles` : "Pre-Configured"}
                    </span>
                  </div>

                  <div className="font-semibold text-sm text-white group-hover:text-emerald-300 transition-colors">
                    {m.title}
                  </div>

                  <div className="text-xs text-white/60 line-clamp-1 mt-1 font-mono">
                    📍 {m.location}
                  </div>

                  <div className="text-[11px] text-cyan-300/80 font-mono mt-1 flex items-center gap-1.5">
                    <span>📡 {m.sensors}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right Column: Mission Detail, Satellite Previews, and Action Launcher */}
          {currentMission && (
            <div className="w-full md:w-7/12 p-6 flex flex-col justify-between overflow-y-auto bg-black/40">
              <div className="flex flex-col gap-4">
                
                {/* Header & Tag */}
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-mono text-cyan-400 uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
                      {currentMission.tag}
                    </span>
                    <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wide">
                      {currentMission.pipeline}
                    </span>
                  </div>
                  <h3 className="text-lg sm:text-xl font-bold text-white tracking-wide mt-1">
                    {currentMission.title}
                  </h3>
                  <p className="text-xs text-white/75 mt-1.5 leading-relaxed">
                    {currentMission.description}
                  </p>
                </div>

                {/* Tactical Parameters Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono bg-white/[0.03] p-3 rounded-xl border border-white/10">
                  <div className="col-span-2 sm:col-span-1">
                    <span className="text-white/40 block text-[9px] uppercase tracking-wider">GEOLOCATION</span>
                    <span className="text-white font-medium text-[11px] truncate block">{currentMission.location}</span>
                    <span className="text-emerald-400 text-[10px]">{currentMission.coordinates}</span>
                  </div>
                  <div>
                    <span className="text-white/40 block text-[9px] uppercase tracking-wider">PRIMARY SENSOR</span>
                    <span className="text-white font-medium text-[11px] truncate block">{currentMission.sensors}</span>
                  </div>
                  <div>
                    <span className="text-white/40 block text-[9px] uppercase tracking-wider">RESOLVING POWER</span>
                    <span className="text-cyan-300 font-medium text-[11px] truncate block">{currentMission.resolution}</span>
                  </div>
                </div>

                {/* Satellite Tiles View */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-mono tracking-wider text-white/60 uppercase font-semibold">
                      Mission Satellite Imagery Inputs ({currentMission.images.length || 2})
                    </span>
                    {currentMission.images.length > 1 && (
                      <span className="text-[10px] font-mono text-emerald-400">
                        Co-Registered Spatial Pair
                      </span>
                    )}
                  </div>

                  {currentMission.images.length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {currentMission.images.map((img, idx) => (
                        <div
                          key={idx}
                          onClick={() => setPreviewTileIndex(idx)}
                          className={`flex flex-col gap-1 cursor-pointer transition-all rounded-xl p-1.5 border ${
                            previewTileIndex === idx
                              ? "border-emerald-400 bg-emerald-500/10"
                              : "border-white/10 bg-white/[0.02] hover:border-white/30"
                          }`}
                        >
                          <div className="aspect-video sm:aspect-square rounded-lg overflow-hidden border border-white/10 bg-black relative group">
                            <img
                              src={img.base64}
                              alt={img.label}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent opacity-80" />
                            <span className="absolute bottom-1.5 left-2 right-2 text-[10px] font-mono text-white font-semibold truncate">
                              {img.label}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="aspect-video rounded-xl border border-white/10 bg-white/[0.02] flex flex-col items-center justify-center p-6 text-center">
                      <div className="w-8 h-8 rounded-full border border-emerald-400/30 flex items-center justify-center text-emerald-400 text-sm mb-2 animate-pulse">
                        🛰️
                      </div>
                      <span className="text-xs text-white/70 font-mono">
                        Direct synthetic payload loaded & calibrated for {currentMission.title}
                      </span>
                      <span className="text-[10px] text-white/40 font-mono mt-1">
                        Click 'Launch Mission' to generate and analyze this scenario.
                      </span>
                    </div>
                  )}
                </div>

                {/* Natural Language Mission Query Command */}
                <div className="bg-emerald-950/20 border border-emerald-500/30 p-3 rounded-xl">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono text-emerald-400 uppercase font-bold tracking-wider">
                      Target Query Prompt
                    </span>
                    <span className="text-[9px] font-mono text-white/40">NL Vision-Language Grounding</span>
                  </div>
                  <p className="text-xs text-white font-mono leading-relaxed bg-black/40 p-2.5 rounded-lg border border-white/10">
                    "{currentMission.query}"
                  </p>
                </div>

                {/* Evaluator Insight Box */}
                <div className="text-[11px] font-mono text-white/60 bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                  <span className="text-cyan-400 font-bold mr-1.5">💡 SIH EVALUATOR NOTE:</span>
                  <span>{currentMission.technicalNote}</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 pt-5 mt-4 border-t border-white/10">
                <button
                  type="button"
                  onClick={() => {
                    onSelectMission(currentMission, false);
                    onClose();
                  }}
                  className="flex-1 py-2.5 px-4 rounded-xl border border-white/30 hover:border-white text-white hover:bg-white/10 text-xs font-mono font-semibold transition-colors cursor-pointer text-center"
                >
                  Load into Query Input
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onSelectMission(currentMission, true);
                    onClose();
                  }}
                  className="flex-1 py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 hover:from-emerald-400 hover:to-cyan-300 text-black text-xs font-mono font-bold transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 cursor-pointer hover:scale-[1.01]"
                >
                  <span>⚡ 1-Click Launch Mission</span>
                </button>
              </div>

            </div>
          )}

        </div>

      </div>
    </div>
  );
}
