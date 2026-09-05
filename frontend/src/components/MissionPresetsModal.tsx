"use client";

import { useState, useEffect } from "react";

export interface MissionImage {
  name: string;
  base64: string;
  label: string;
}

export interface SampleMission {
  id: string;
  title: string;
  tag: string;
  location: string;
  sensors: string;
  query: string;
  description: string;
  images: MissionImage[];
}

interface MissionPresetsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectMission: (mission: SampleMission, autoSubmit?: boolean) => void;
}

export default function MissionPresetsModal({
  isOpen,
  onClose,
  onSelectMission
}: MissionPresetsModalProps) {
  const [missions, setMissions] = useState<SampleMission[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

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
            setMissions(data.missions);
            setSelectedId(data.missions[0].id);
          }
        }
      } catch (err) {
        console.error("Failed to load sample missions", err);
      } finally {
        setLoading(false);
      }
    };

    fetchMissions();
  }, [isOpen]);

  if (!isOpen) return null;

  const currentMission = missions.find(m => m.id === selectedId) || missions[0];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-[#0e0e12] border border-white/20 rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-slide-up">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <span className="text-xl">🛰️</span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono tracking-widest text-emerald-400 font-bold uppercase">
                  ISRO / SAC — PS 26167
                </span>
                <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 text-[10px] font-mono">
                  Curated Presets
                </span>
              </div>
              <h2 className="text-lg font-bold text-white tracking-wide">
                Tactical Earth Observation Mission Scenarios
              </h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full border border-white/20 flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        {loading ? (
          <div className="flex-1 flex items-center justify-center p-12 text-white/60 font-mono text-sm">
            <div className="animate-spin mr-3 w-5 h-5 border-2 border-emerald-400 border-t-transparent rounded-full" />
            Loading ISRO Mission Scenarios...
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col md:flex-row divide-y md:divide-y-0 md:divide-x divide-white/10">
            {/* Left: Mission Selector List */}
            <div className="w-full md:w-5/12 p-4 flex flex-col gap-2.5 overflow-y-auto">
              <span className="text-[10px] font-mono tracking-widest text-white/40 uppercase px-2">
                SELECT OPERATIONAL SCENARIO
              </span>
              {missions.map(m => {
                const isSelected = m.id === selectedId;
                return (
                  <button
                    key={m.id}
                    onClick={() => setSelectedId(m.id)}
                    className={`text-left p-3 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-white/10 border-emerald-400/80 shadow-lg shadow-emerald-500/10"
                        : "bg-white/[0.02] border-white/10 hover:bg-white/[0.06] hover:border-white/20"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className="text-[10px] font-mono font-bold tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                        {m.tag}
                      </span>
                      <span className="text-[10px] text-white/40 font-mono">
                        {m.images.length} {m.images.length > 1 ? "Tiles" : "Tile"}
                      </span>
                    </div>
                    <div className="font-semibold text-sm text-white">{m.title}</div>
                    <div className="text-xs text-white/60 line-clamp-1 mt-0.5">{m.location}</div>
                  </button>
                );
              })}
            </div>

            {/* Right: Mission Detail & Instant Preview */}
            {currentMission && (
              <div className="w-full md:w-7/12 p-6 flex flex-col justify-between overflow-y-auto bg-black/40">
                <div className="flex flex-col gap-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[11px] font-mono text-cyan-400 uppercase font-semibold">
                        {currentMission.tag}
                      </span>
                    </div>
                    <h3 className="text-xl font-bold text-white tracking-wide">
                      {currentMission.title}
                    </h3>
                    <p className="text-xs text-white/70 mt-1 leading-relaxed">
                      {currentMission.description}
                    </p>
                  </div>

                  {/* Metadata Grid */}
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-white/[0.03] p-3 rounded-xl border border-white/10">
                    <div>
                      <span className="text-white/40 block text-[10px]">GEOLOCATION</span>
                      <span className="text-white font-medium">{currentMission.location}</span>
                    </div>
                    <div>
                      <span className="text-white/40 block text-[10px]">PRIMARY SENSOR</span>
                      <span className="text-white font-medium">{currentMission.sensors}</span>
                    </div>
                  </div>

                  {/* Satellite Tile Previews */}
                  <div>
                    <span className="text-[10px] font-mono tracking-wider text-white/50 block mb-2 uppercase">
                      Co-Registered Satellite Imagery Inputs ({currentMission.images.length})
                    </span>
                    <div className="grid grid-cols-2 gap-3">
                      {currentMission.images.map((img, idx) => (
                        <div key={idx} className="flex flex-col gap-1">
                          <div className="aspect-square rounded-lg overflow-hidden border border-white/20 bg-black relative group">
                            <img
                              src={img.base64}
                              alt={img.label}
                              className="w-full h-full object-cover"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent opacity-80" />
                            <span className="absolute bottom-1.5 left-2 right-2 text-[10px] font-mono text-white/90 truncate">
                              {img.label}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Query Prompt Box */}
                  <div className="bg-white/[0.04] p-3 rounded-xl border border-white/10">
                    <span className="text-[10px] font-mono text-emerald-400 block mb-1 uppercase font-semibold">
                      Mission Query Command
                    </span>
                    <p className="text-xs text-white font-mono leading-relaxed">
                      "{currentMission.query}"
                    </p>
                  </div>
                </div>

                {/* Footer Action Buttons */}
                <div className="flex gap-3 pt-6 mt-4 border-t border-white/10">
                  <button
                    type="button"
                    onClick={() => {
                      onSelectMission(currentMission, false);
                      onClose();
                    }}
                    className="flex-1 py-2.5 px-4 rounded-xl border border-white/30 text-white hover:bg-white/10 text-xs font-mono font-semibold transition-colors"
                  >
                    Load into Workspace
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      onSelectMission(currentMission, true);
                      onClose();
                    }}
                    className="flex-1 py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-black text-xs font-mono font-bold transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
                  >
                    <span>⚡ Launch Mission</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
