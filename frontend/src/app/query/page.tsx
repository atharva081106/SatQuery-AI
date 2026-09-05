"use client";

import { useState, useRef, useEffect, lazy, Suspense } from "react";

const GeoJSONMapOverlay = lazy(() => import('@/components/GeoJSONMapOverlay'));

type Message = {
  role: "user" | "assistant";
  content: string;
  result?: any;
};

import { useAuth } from "@/context/AuthContext";

import BackgroundSlideshow from '@/components/BackgroundSlideshow';
import FadeInScroll from '@/components/FadeInScroll';
import FramerGlobe from '@/components/FramerGlobe';
import jsPDF from 'jspdf';
import SwipeSlider from '@/components/SwipeSlider';
import MapExplorer from '@/components/MapExplorer';
import TiffPreview from '@/components/TiffPreview';
import MissionPresetsModal, { SampleMission } from '@/components/MissionPresetsModal';

function renderInline(text: string) {
  if (!text) return null;
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <strong key={i} className="text-white font-semibold">
              {part.slice(2, -2)}
            </strong>
          );
        }
        return part;
      })}
    </>
  );
}

function renderTable(tableLinesText: string, key: any) {
  const rawLines = tableLinesText.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  const tableLines = rawLines.filter(l => l.includes("|") && !/^\|?\s*[-:]+[-| :]*\|?$/.test(l));
  
  if (tableLines.length === 0) return null;

  const headerCells = tableLines[0]
    .split("|")
    .map(c => c.trim())
    .filter(c => c.length > 0);

  const rowLines = tableLines.slice(1);
  const rows = rowLines.map(line =>
    line
      .split("|")
      .map(c => c.trim())
      .filter(c => c.length > 0)
  );

  return (
    <div key={key} className="overflow-x-auto my-2 rounded-xl border border-white/20 bg-white/[0.04] shadow-sm">
      <table className="w-full text-left border-collapse text-xs sm:text-sm">
        <thead>
          <tr className="border-b border-white/20 bg-white/10 text-white font-medium">
            {headerCells.map((h, idx) => (
              <th
                key={idx}
                className={`py-2.5 px-4 font-mono tracking-wider text-[11px] sm:text-xs uppercase text-white/80 ${
                  idx === headerCells.length - 1 ? 'text-right' : ''
                }`}
              >
                {renderInline(h)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {rows.map((row, rIdx) => {
            const isTotal = row.some(cell => cell.toLowerCase().includes("total"));
            return (
              <tr
                key={rIdx}
                className={
                  isTotal
                    ? "bg-white/10 font-semibold text-white border-t border-white/20"
                    : "hover:bg-white/[0.05] transition-colors text-white/90"
                }
              >
                {row.map((cell, cIdx) => (
                  <td
                    key={cIdx}
                    className={`py-2.5 px-4 ${
                      cIdx === row.length - 1 ? 'text-right font-mono text-emerald-300 font-medium' : ''
                    } ${isTotal ? 'text-white' : ''}`}
                  >
                    {renderInline(cell)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FormattedMessage({ content }: { content: string }) {
  if (!content) return null;

  // Split into blocks by double newline
  const blocks = content.split(/\n\n+/).map(b => b.trim()).filter(b => b.length > 0);

  return (
    <div className="flex flex-col gap-4 text-sm leading-relaxed">
      {blocks.map((block, bIdx) => {
        // 1. Alert block (Compatibility Failure or warning)
        if (block.includes("COMPATIBILITY CHECK FAILED") || block.includes("Spatial Disparity Detected") || block.startsWith("⚠️")) {
          const lines = block.split("\n").map(l => l.trim()).filter(l => l.length > 0);
          const alertTitle = lines[0];
          const remainingLines = lines.slice(1);
          return (
            <div key={bIdx} className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex flex-col gap-2 shadow-inner">
              <div className="flex items-center gap-2 text-amber-400 font-mono text-xs tracking-wider uppercase font-bold">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                <span>{alertTitle.replace(/^⚠️\s*/, "")}</span>
              </div>
              {remainingLines.map((line, lIdx) => (
                <p key={lIdx} className="text-white/85 text-xs sm:text-sm font-sans leading-relaxed">
                  {renderInline(line)}
                </p>
              ))}
            </div>
          );
        }

        // 2. Table block
        const blockLines = block.split("\n").map(l => l.trim()).filter(l => l.length > 0);
        const tableStartIndex = blockLines.findIndex(l => l.includes("|"));
        if (tableStartIndex !== -1 && blockLines.filter(l => l.includes("|")).length >= 2) {
          const prefixLines = blockLines.slice(0, tableStartIndex);
          const tableLines = blockLines.slice(tableStartIndex);
          return (
            <div key={bIdx} className="flex flex-col gap-2">
              {prefixLines.length > 0 && (
                <p className="text-white/90 text-sm leading-relaxed">
                  {renderInline(prefixLines.join(" "))}
                </p>
              )}
              {renderTable(tableLines.join("\n"), `${bIdx}-tbl`)}
            </div>
          );
        }

        // 3. Bullet list section
        const hasBullets = blockLines.some(l => l.startsWith("•") || l.startsWith("-"));
        if (hasBullets) {
          return (
            <div key={bIdx} className="flex flex-col gap-2">
              {blockLines.map((line, lIdx) => {
                if (line.startsWith("•") || line.startsWith("-")) {
                  const bulletText = line.replace(/^[•\-]\s*/, "");
                  return (
                    <div key={lIdx} className="flex items-start gap-2.5 pl-1 text-xs sm:text-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0"></span>
                      <span className="text-white/90 leading-relaxed">{renderInline(bulletText)}</span>
                    </div>
                  );
                } else {
                  return (
                    <div key={lIdx} className="text-white font-medium text-sm flex items-center gap-1.5 mt-1">
                      {renderInline(line)}
                    </div>
                  );
                }
              })}
            </div>
          );
        }

        // 4. Standard paragraph
        return (
          <p key={bIdx} className="text-white/90 text-sm leading-relaxed">
            {renderInline(block)}
          </p>
        );
      })}
    </div>
  );
}

export default function Home() {
  const {
    canQuery,
    incrementQueryCount,
    openAuthModal,
    queryCount,
    maxFreeQueries,
    isAuthenticated,
    user,
    logout,
  } = useAuth();

  const [images, setImages] = useState<File[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [showPresets, setShowPresets] = useState(false);
  const presetsRef = useRef<HTMLDivElement>(null);
  const [expandedImage, setExpandedImage] = useState<string | null>(null);
  const [showMapExplorer, setShowMapExplorer] = useState(false);
  const [showMissionPresets, setShowMissionPresets] = useState(false);

  const dataURLtoFile = (dataurl: string, filename: string): File => {
    const arr = dataurl.split(',');
    const mimeMatch = arr[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : 'image/png';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  const handleDownloadGeoJSON = () => {
    if (!latestResult?.geojson_data) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(latestResult.geojson_data, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `SatQuery_GIS_Vector_${Date.now()}.geojson`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const queryCategories = [
    {
      category: "TERRAIN & LAND COVER ANALYSIS",
      queries: [
        "Classify land cover breakdown",
        "How much green cover and vegetation is there?",
        "Are there any water bodies or rivers in this area?",
        "Identify buildings, roads, and settlements",
        "Give a plain-English overview of this scene"
      ]
    },
    {
      category: "CHANGE DETECTION (BEFORE & AFTER)",
      queries: [
        "What changed between these two dates?",
        "Measure new construction and building expansion",
        "Check for deforestation and tree loss",
        "Analyze flood extent and submerged land"
      ]
    },
    {
      category: "FIND & HIGHLIGHT FEATURES (GROUNDING)",
      queries: [
        "Highlight all water bodies and lakes",
        "Locate buildings and storage facilities",
        "Pinpoint roads and transport networks",
        "Find green parks and farmland"
      ]
    },
    {
      category: "CLOUD PENETRATION (RADAR / SAR)",
      queries: [
        "See through clouds using radar (SAR)",
        "Identify water bodies hidden under clouds",
        "Detect ships and structures beneath cloud cover"
      ]
    }
  ];

  useEffect(() => {
    // Check for acquired image from Map Explorer
    const acquiredBase64 = sessionStorage.getItem("satquery_acquired_image");
    if (acquiredBase64) {
      // Convert base64 back to File
      fetch(acquiredBase64)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], "acquired_satellite_image.png", { type: "image/png" });
          setImages(prev => {
            // Add a welcome message to prompt the user (only if not already there)
            if (!prev.some(f => f.name === "acquired_satellite_image.png") && prev.length === 0) {
              setMessages([{ 
                role: "assistant", 
                content: "I have successfully acquired and loaded your satellite imagery from the Map Explorer. The target area is ready for analysis. What would you like me to look for?"
              }]);
            }
            
            // Generate a unique filename if adding multiple acquired images
            const uniqueName = `acquired_satellite_image_${Date.now()}.png`;
            const file = new File([blob], uniqueName, { type: "image/png" });

            return [...prev, file].slice(0, 2); // Max 2 images
          });
          // Clean up
          sessionStorage.removeItem("satquery_acquired_image");
          sessionStorage.removeItem("satquery_acquired_bbox");
        });
    }
  }, []);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (presetsRef.current && !presetsRef.current.contains(event.target as Node)) {
        setShowPresets(false);
      }
    }
    if (showPresets) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showPresets]);

  const loadingStages = [
    "INITIALIZING PIPELINE...",
    "ANALYZING MULTIMODAL CONTEXT...",
    "EXTRACTING SPATIAL FEATURES...",
    "SYNTHESIZING RESPONSE...",
  ];
  const [loadingTextIndex, setLoadingTextIndex] = useState(0);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (loading) {
      interval = setInterval(() => {
        setLoadingTextIndex((prev) => (prev + 1) % loadingStages.length);
      }, 1500);
    } else {
      setLoadingTextIndex(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setImages((prev) => [...prev, ...newFiles].slice(0, 2));
    }
  };

  const removeImage = (indexToRemove: number) => {
    setImages(images.filter((_, index) => index !== indexToRemove));
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files) {
      const newFiles = Array.from(e.dataTransfer.files);
      setImages((prev) => [...prev, ...newFiles].slice(0, 2));
    }
  };

  const executeAnalysis = async (queryText: string, imageFiles: File[]) => {
    if (!canQuery) {
      openAuthModal("Sign in to continue using SatQuery AI. You have used your 3 free satellite analyses.");
      return;
    }

    if (!queryText) {
      setError("Please provide a query.");
      return;
    }
    if (imageFiles.length === 0 && messages.length === 0) {
      setError("Please provide at least one image to start the analysis.");
      return;
    }

    setLoading(true);
    setError(null);

    const newMessages = [...messages, { role: "user", content: queryText } as Message];
    setMessages(newMessages);
    setQuery("");

    const formData = new FormData();
    formData.append("query", queryText);

    const historyPayload = messages.map(m => ({ 
      role: m.role, 
      user: m.role === 'user' ? m.content : '', 
      assistant: m.role === 'assistant' ? m.content : '' 
    }));
    formData.append("history", JSON.stringify(historyPayload));

    imageFiles.forEach((img) => {
      formData.append("images", img);
    });

    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${backendUrl}/api/query`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process request");
      }

      const data = await response.json();
      if (data.status === "error") {
        throw new Error(data.message);
      }
      
      setMessages([...newMessages, { role: "assistant", content: data.answer, result: data }]);
      incrementQueryCount();

      // Trigger authentication popup right after 3 free queries are completed
      if (!isAuthenticated && (queryCount + 1) >= maxFreeQueries) {
        setTimeout(() => {
          openAuthModal("Sign in to continue using SatQuery AI. You have used your 3 free satellite analyses.");
        }, 1000);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
      setMessages([...newMessages, { role: "assistant", content: "Sorry, I encountered an error: " + err.message }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await executeAnalysis(query, images);
  };

  const handleSelectMission = (mission: SampleMission, autoSubmit = false) => {
    if (autoSubmit && !canQuery) {
      openAuthModal("Sign in to continue using SatQuery AI. You have used your 3 free satellite analyses.");
      return;
    }
    const files: File[] = [];
    for (const img of mission.images) {
      const f = dataURLtoFile(img.base64, img.name);
      files.push(f);
    }
    setImages(files);
    setQuery(mission.query);
    setError(null);
    if (autoSubmit) {
      executeAnalysis(mission.query, files);
    }
  };

    const latestAssistantMessage = [...messages].reverse().find(m => m.role === 'assistant' && m.result);
  const latestResult = latestAssistantMessage?.result;

  const handleDownloadReport = () => {
    // Collect all queries and their corresponding results
    const allResults = messages.reduce<{query: string, result: any}[]>((acc, msg, idx, arr) => {
      if (msg.role === 'assistant' && msg.result) {
        let queryStr = "N/A";
        for (let i = idx - 1; i >= 0; i--) {
          if (arr[i].role === 'user') {
            queryStr = arr[i].content;
            break;
          }
        }
        acc.push({ query: queryStr, result: msg.result });
      }
      return acc;
    }, []);

    if (allResults.length === 0) return;

    try {
      const doc = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
      });

      // Dark header banner
      doc.setFillColor(12, 12, 16);
      doc.rect(0, 0, 210, 36, "F");

      // Title
      doc.setFont("helvetica", "bold");
      doc.setFontSize(15);
      doc.setTextColor(255, 255, 255);
      doc.text("SATQUERY AI - REMOTE SENSING AUDIT REPORT", 14, 15);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(8.5);
      doc.setTextColor(170, 170, 185);
      doc.text("OPERATIONAL MISSION AUDIT | ISRO/SAC & BENCHMARK SUITE", 14, 22);
      doc.text(`TIMESTAMP: ${new Date().toISOString()}`, 14, 28);
      
      let y = 46;

      allResults.forEach((item, index) => {
        const { query, result } = item;
        const summary = result.execution_summary || {};

        if (index > 0) {
          doc.addPage();
          y = 20;
        }

        // Query Header
        doc.setFont("helvetica", "bold");
        doc.setFontSize(11);
        doc.setTextColor(20, 20, 25);
        doc.text(`QUERY ${index + 1}: ${query}`, 14, y);
        y += 10;

        // 1. Mission Overview Table/Box
        doc.setFillColor(246, 246, 249);
        doc.roundedRect(14, y, 182, 36, 2, 2, "F");
        doc.setDrawColor(215, 215, 222);
        doc.roundedRect(14, y, 182, 36, 2, 2, "D");

        doc.setFont("helvetica", "bold");
        doc.setFontSize(9.5);
        doc.setTextColor(20, 20, 25);
        doc.text("1. MISSION OVERVIEW & TASK ROUTING", 18, y + 7);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8.5);
        doc.setTextColor(60, 60, 70);
        doc.text(`Selected Task: ${summary.selected_task || "N/A"}`, 18, y + 14);
        doc.text(`Specialist Tool: ${summary.tool_used || "N/A"}`, 18, y + 20);
        doc.text(`Model Provenance: ${summary.model_provenance || "SatQuery-RS-Adapted-v1.2 (Fine-tuned)"}`, 18, y + 26);
        doc.text(`Input Configuration: ${summary.input_scope || "Standard Input"}`, 18, y + 32);

        y += 44;

        // 2. Input Compatibility & Coherence Box
        const isFailed = result.compatibility_status === "FAILED";
        if (isFailed) {
          doc.setFillColor(254, 242, 242);
          doc.setDrawColor(248, 113, 113);
        } else {
          doc.setFillColor(240, 253, 244);
          doc.setDrawColor(74, 222, 128);
        }
        doc.roundedRect(14, y, 182, 28, 2, 2, "F");
        doc.roundedRect(14, y, 182, 28, 2, 2, "D");

        doc.setFont("helvetica", "bold");
        doc.setFontSize(9.5);
        if (isFailed) {
          doc.setTextColor(185, 28, 28);
          doc.text("2. INPUT COMPATIBILITY: REJECTED (SPATIAL DISPARITY DETECTED)", 18, y + 7);
        } else {
          doc.setTextColor(21, 128, 61);
          doc.text("2. INPUT COMPATIBILITY: VERIFIED (SPATIAL CO-REGISTRATION CONFIRMED)", 18, y + 7);
        }

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8.5);
        doc.setTextColor(50, 50, 60);
        const coherence = ((result.spatial_coherence_score ?? 1.0) * 100).toFixed(1);
        const confidence = ((result.confidence ?? 0.9) * 100).toFixed(1);
        doc.text(`Spatial Coherence: ${coherence}% | Confidence Score: ${confidence}%`, 18, y + 14);
        doc.text(
          isFailed 
            ? "Warning: Images belong to non-overlapping geographic regions. Change analysis halted to prevent hallucinated changes." 
            : "Integrity: High keypoint spatial correlation and coordinate alignment verified across inputs.",
          18, 
          y + 21
        );

        y += 36;

        // 3. Agentic Reasoning Trace
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9.5);
        doc.setTextColor(20, 20, 25);
        doc.text("3. AGENTIC CONTROLLER REASONING TRACE", 14, y);
        y += 5;

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(70, 70, 80);
        const reasoningLines = doc.splitTextToSize(summary.agent_reasoning || "Standard routing execution.", 182);
        doc.text(reasoningLines, 14, y);
        y += (reasoningLines.length * 4) + 6;

        // 4. Analysis Synthesis & Response
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9.5);
        doc.setTextColor(20, 20, 25);
        doc.text("4. SYSTEM SYNTHESIS & INFERENCE", 14, y);
        y += 5;

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(40, 40, 50);
        const answerClean = (result.answer || "N/A").replace(/\*\*/g, "").replace(/\n\n/g, "\n");
        const answerLines = doc.splitTextToSize(answerClean, 182);
        doc.text(answerLines, 14, y);
        y += (answerLines.length * 4) + 8;

        // 5. Visual Evidence (if available and base64)
        const evidence = result.visual_evidence?.[0];
        if (evidence && evidence.image_base64 && evidence.image_base64.startsWith("data:image")) {
          try {
            if (y > 210) {
              doc.addPage();
              y = 20;
            }
            doc.setFont("helvetica", "bold");
            doc.setFontSize(9.5);
            doc.setTextColor(20, 20, 25);
            doc.text(`5. VISUAL EVIDENCE: ${evidence.description || "Analytical Canvas"}`, 14, y);
            y += 4;
            
            doc.addImage(evidence.image_base64, "PNG", 14, y, 115, 65);
            y += 70;
          } catch (imgErr) {
            console.error("Could not embed evidence image into PDF:", imgErr);
          }
        }
      });

      // Page Footers
      const pageCount = doc.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(7.5);
        doc.setTextColor(150, 150, 160);
        doc.text("CONFIDENTIAL & PROPRIETARY — SATQUERY AI CORE ENGINE | ISRO/SAC EVALUATION PROTOCOL", 14, 290);
        doc.text(`PAGE ${i} OF ${pageCount}`, 182, 290);
      }

      doc.save(`SatQuery_Audit_Report_${Date.now()}.pdf`);
    } catch (err) {
      console.error("Failed to generate PDF:", err);
      // Fallback to text file if browser canvas fails
      const blob = new Blob([JSON.stringify(allResults, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `SatQuery_Audit_Report_${Date.now()}.json`;
      link.click();
    }
  };

  return (
    <main className="h-screen bg-black relative overflow-hidden flex flex-col">
      
      {/* NATURAL COLOUR GLOBE BACKGROUND (EMPTY STATE) */}
      {!messages.some(m => m.role === 'user') && (
        <div className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none">
          <div className="relative z-[6] pointer-events-auto w-[600px] h-[600px] rounded-full flex items-center justify-center">
            <div className="w-full h-full opacity-100">
              <FramerGlobe />
            </div>
          </div>
        </div>
      )}

      {/* FIXED TOP NAV OVERLAY */}
      <nav className="w-full flex justify-between items-center px-6 lg:px-8 py-4 z-50 transition-all">
        <div className="flex items-center gap-3">
          <a href="/" className="display-lg tracking-widest text-white hover:opacity-70 transition-opacity pointer-events-auto flex items-center gap-2">
            <span>SATQUERY AI.</span>
          </a>
          <span className="hidden sm:inline-block text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 font-mono font-bold tracking-wider border border-emerald-500/30">
            ISRO / SAC — PS 26167
          </span>
        </div>

        <div className="flex gap-2.5 sm:gap-3 items-center">
          {/* Guest Sign In / Sign Up Trigger */}
          {!isAuthenticated && (
            <button
              type="button"
              onClick={() => openAuthModal("Sign in to sync mission logs and access unlimited high-resolution analyses.", "signin")}
              className="micro-cap border border-white/20 hover:border-white text-white/80 hover:text-white hover:bg-white/10 px-3 py-1.5 rounded-full transition-all duration-200 cursor-pointer pointer-events-auto tracking-widest uppercase text-[11px]"
              title="Sign in to your account"
            >
              SIGN IN
            </button>
          )}

          {/* Authenticated User Callsign */}
          {isAuthenticated && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono font-bold tracking-widest uppercase flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>{user?.name?.toUpperCase() || "PRO OPERATOR"}</span>
              </span>
              <button
                type="button"
                onClick={logout}
                className="text-[10px] text-white/50 hover:text-white uppercase tracking-wider underline cursor-pointer"
                title="Sign out"
              >
                Sign Out
              </button>
            </div>
          )}

          {/* Demo Queries Launcher (SpaceX Aerospace Aesthetic) */}
          <button
            type="button"
            onClick={() => setShowMissionPresets(true)}
            className="micro-cap border border-white/30 hover:border-white text-white hover:bg-white hover:text-black px-4 py-1.5 rounded-full transition-all duration-200 cursor-pointer pointer-events-auto tracking-widest uppercase font-bold"
            title="Explore & Launch Pre-Configured Demo Queries"
          >
            DEMO QUERIES
          </button>

          <a href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 px-3.5 py-1.5 rounded-full flex items-center gap-1.5 pointer-events-auto">
            <span>&larr;</span> <span className="hidden md:inline">DASHBOARD</span>
          </a>
        </div>
      </nav>

      {/* MAIN APPLICATION CONTAINER */}
      <div className={`flex-1 flex flex-col lg:flex-row w-full mx-auto p-4 lg:p-8 gap-8 z-10 pb-8 min-h-0 transition-all duration-500 pointer-events-none ${latestResult ? 'max-w-[1500px]' : 'max-w-4xl'}`}>
        
        {/* LEFT/MAIN: CHAT & INPUT */}
        <div className="flex-1 flex flex-col justify-end relative min-h-0">
          
          <div className={`overflow-y-auto mb-4 custom-scrollbar pr-4 flex flex-col gap-6 w-full min-h-0 relative z-10 ${messages.length > 0 ? 'pointer-events-auto' : ''}`}>
            {messages.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center pointer-events-none h-full mt-32">
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className="micro-cap text-white/50 mb-1">
                  {msg.role === 'user' ? 'COMMAND' : 'SYSTEM'}
                </div>
                <div className={`text-sm max-w-[85%] p-4 rounded-xl border shadow-lg pointer-events-auto ${
                  msg.role === 'user' 
                    ? 'bg-white/5 text-white border-white/40 font-mono text-xs whitespace-pre-wrap' 
                    : 'bg-black/90 text-white border-[#3a3a3f]'
                }`}>
                  {msg.role === 'user' ? msg.content : <FormattedMessage content={msg.content} />}
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="flex flex-col items-start mt-2">
                <div className="micro-cap text-white/50 mb-2">SYSTEM</div>
                <div className="flex items-center gap-3 bg-black/60 backdrop-blur-sm px-4 py-3 border border-[#3a3a3f] rounded-lg">
                  <div className="w-3.5 h-3.5 rounded-full border-2 border-white/20 border-t-white animate-spin"></div>
                  <div className="micro-cap text-white animate-pulse">
                    {loadingStages[loadingTextIndex]}
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="w-full relative z-50 pointer-events-auto">
            {error && (
              <div className="micro-cap text-[#ff3000] mb-3 uppercase">
                ERROR: {error}
              </div>
            )}
            
            <form onSubmit={handleSubmit} className="flex flex-col gap-3">
              
              {/* IMAGE PREVIEWS */}
              {images.length > 0 && (
                <div className="flex gap-2 flex-wrap mb-2">
                  {images.map((img, i) => {
                    const isTiff = img.name.toLowerCase().endsWith('.tif') || img.name.toLowerCase().endsWith('.tiff');
                    const objUrl = !isTiff ? URL.createObjectURL(img) : "";
                    return (
                      <div key={i} className="w-16 h-16 relative group rounded-md overflow-hidden border border-[#3a3a3f]">
                        {isTiff ? (
                          <TiffPreview 
                            file={img} 
                            className="w-full h-full" 
                          />
                        ) : (
                          <img 
                            src={objUrl} 
                            alt={`Preview ${i}`} 
                            className="w-full h-full object-cover cursor-pointer hover:opacity-70 transition-opacity" 
                            onClick={() => setExpandedImage(objUrl)}
                          />
                        )}
                        <button 
                          type="button" 
                          className="absolute top-1 right-1 bg-black/80 text-white w-5 h-5 rounded-full flex items-center justify-center text-[10px] opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500"
                          onClick={() => removeImage(i)}
                          title="Remove image"
                        >✕</button>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* UNIFIED INPUT BAR */}
              <div 
                className={`relative flex items-center bg-black/60 backdrop-blur-md border border-[#3a3a3f] rounded-2xl p-1.5 transition-colors ${isDragOver ? 'bg-white/10 border-white/50' : ''}`}
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
              >
                {/* Hidden File Input */}
                <input
                  type="file"
                  multiple
                  className="hidden"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  accept="image/*, .tif, .tiff, .geotiff, .jp2, .j2k, .webp, .png, .jpg, .jpeg, .bmp, .avif, .fits, .fit"
                  id="file-upload"
                />
                
                {/* Attachment Button */}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-8 h-8 flex items-center justify-center text-white/50 hover:text-white transition-colors rounded-full hover:bg-white/10 shrink-0 ml-1"
                  title="Attach Images"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                  </svg>
                </button>

                {/* Map Acquire Button */}
                <button
                  type="button"
                  onClick={() => setShowMapExplorer(true)}
                  className="w-8 h-8 flex items-center justify-center text-white/50 hover:text-white transition-colors rounded-full hover:bg-white/10 shrink-0"
                  title="Acquire Imagery from Map"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon>
                    <line x1="9" y1="3" x2="9" y2="18"></line>
                    <line x1="15" y1="6" x2="15" y2="21"></line>
                  </svg>
                </button>

                {/* Presets Button beside Attachment Button */}
                <div className="relative shrink-0 flex items-center ml-1 mr-1" ref={presetsRef}>
                  <button
                    type="button"
                    onClick={() => setShowPresets(!showPresets)}
                    className={`h-7 px-2.5 rounded-full border transition-all flex items-center gap-1.5 text-[9px] font-mono tracking-wider cursor-pointer ${
                      showPresets 
                        ? 'border-white bg-white text-black font-bold shadow-lg' 
                        : 'border-[#3a3a3f] bg-white/5 text-white/60 hover:text-white hover:border-white/40 hover:bg-white/10'
                    }`}
                    title="Preset Remote-Sensing Queries"
                  >
                    <span className={showPresets ? "text-black" : "text-white/80"}>✦</span>
                    <span className="hidden sm:inline uppercase">QUERIES</span>
                    <span className={`text-[7px] transition-transform duration-200 ${showPresets ? 'rotate-180' : ''}`}>▼</span>
                  </button>

                  {/* Expandable Query Popover Library */}
                  {showPresets && (
                    <div 
                      className="absolute bottom-full mb-3 left-0 w-[92vw] sm:w-[560px] max-h-[380px] overflow-y-auto custom-scrollbar bg-black/95 backdrop-blur-xl border border-[#3a3a3f] rounded-xl p-4 shadow-2xl z-50 animate-slide-up pointer-events-auto touch-pan-y"
                      onWheel={(e) => e.stopPropagation()}
                    >
                      <div className="flex justify-between items-center pb-3 mb-3 border-b border-[#2a2a2f]">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                          <span className="micro-cap text-white font-semibold">REMOTE SENSING QUERY LIBRARY</span>
                        </div>
                        <button 
                          type="button" 
                          onClick={() => setShowPresets(false)}
                          className="text-white/40 hover:text-white text-xs px-2 py-0.5 rounded border border-transparent hover:border-white/20 cursor-pointer transition-colors"
                        >
                          ✕
                        </button>
                      </div>

                      <div className="flex flex-col gap-4">
                        {queryCategories.map((cat, idx) => (
                          <div key={idx} className="flex flex-col gap-1.5">
                            <div className="text-[9px] tracking-widest text-white/40 font-mono font-semibold uppercase">
                              {cat.category}
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {cat.queries.map((q, qIdx) => (
                                <button
                                  key={qIdx}
                                  type="button"
                                  onClick={() => {
                                    setQuery(q);
                                    setShowPresets(false);
                                  }}
                                  className="text-left text-xs px-3 py-1.5 rounded-md bg-white/5 hover:bg-white text-white/80 hover:text-black transition-all border border-white/10 hover:border-white cursor-pointer"
                                >
                                  {q}
                                </button>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Text Input */}
                <input
                  type="text"
                  className="flex-1 bg-transparent border-none outline-none text-white px-3 py-2 text-sm placeholder-white/40"
                  placeholder="ENTER COMMAND OR NATURAL LANGUAGE QUERY..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={loading}
                />

                {/* Execute Button */}
                <button 
                  type="submit" 
                  disabled={loading || (!query && images.length === 0)}
                  className="ml-2 h-8 px-5 rounded-full bg-white text-black font-semibold text-[10px] tracking-widest uppercase hover:bg-white/80 disabled:opacity-30 disabled:cursor-not-allowed transition-all shrink-0"
                >
                  Execute
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* RIGHT/TRACE: ONLY SHOWN ON RESULT */}
        {latestResult && (
            <div className="w-full lg:w-1/3 bg-black/60 backdrop-blur-md border border-[#3a3a3f] rounded-lg p-6 flex flex-col h-full overflow-y-auto custom-scrollbar shadow-2xl animate-slide-up pointer-events-auto">
              <div className="flex justify-between items-center mb-6 pb-4 border-b border-[#2a2a2f]">
                <h3 className="display-lg flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                  TRACE
                </h3>
                <button
                  onClick={handleDownloadReport}
                  className="micro-cap px-3 py-1.5 rounded-full border border-white/30 text-white hover:bg-white hover:text-black transition-all flex items-center gap-1 text-[10px] cursor-pointer pointer-events-auto"
                  title="Export Official Mission PDF Audit Report"
                >
                  <span>&darr;</span> AUDIT PDF
                </button>
              </div>

              {/* COMPATIBILITY STATUS BADGE */}
              <div className="mb-6">
                <div className="micro-cap text-white/50 mb-1.5">INPUT COMPATIBILITY CHECK</div>
                {latestResult.compatibility_status === "FAILED" ? (
                  <div className="bg-[#ff3000]/10 border border-[#ff3000]/50 rounded p-3 flex flex-col gap-1">
                    <span className="text-[#ff3000] font-semibold text-xs tracking-wider uppercase flex items-center gap-1.5">
                      <span>⚠</span> COMPATIBILITY REJECTED
                    </span>
                    <span className="text-white/70 text-[11px] leading-tight">
                      Spatial Disparity Detected: Images belong to non-overlapping regions or lack co-registration.
                    </span>
                  </div>
                ) : (
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded p-2.5 flex items-center gap-2">
                    <span className="text-emerald-400 text-xs font-bold">✓</span>
                    <span className="text-emerald-300 font-semibold text-xs tracking-wider uppercase">
                      VERIFIED CO-REGISTERED PAIR
                    </span>
                  </div>
                )}
              </div>

              {/* CONFIDENCE & COHERENCE METRICS */}
              <div className="mb-6 grid grid-cols-2 gap-3 bg-[#0a0a0a] border border-[#2a2a2f] p-3 rounded-lg">
                <div>
                  <div className="micro-cap text-white/50 text-[10px] mb-1">CONFIDENCE</div>
                  <div className="text-lg font-light tracking-tight text-white">
                    {((latestResult.confidence ?? 0.9) * 100).toFixed(0)}%
                  </div>
                  <div className="w-full bg-[#2a2a2f] h-1.5 rounded-full mt-1 overflow-hidden">
                    <div 
                      className={`h-full ${latestResult.compatibility_status === 'FAILED' ? 'bg-[#ff3000]' : 'bg-white'}`}
                      style={{ width: `${Math.min(100, Math.max(5, (latestResult.confidence ?? 0.9) * 100))}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="micro-cap text-white/50 text-[10px] mb-1">SPATIAL COHERENCE</div>
                  <div className="text-lg font-light tracking-tight text-white">
                    {((latestResult.spatial_coherence_score ?? 1.0) * 100).toFixed(0)}%
                  </div>
                  <div className="w-full bg-[#2a2a2f] h-1.5 rounded-full mt-1 overflow-hidden">
                    <div 
                      className={`h-full ${latestResult.spatial_coherence_score < 0.28 ? 'bg-[#ff3000]' : 'bg-white'}`}
                      style={{ width: `${Math.min(100, Math.max(5, (latestResult.spatial_coherence_score ?? 1.0) * 100))}%` }}
                    />
                  </div>
                </div>
              </div>
            
            <div className="mb-6">
              <div className="micro-cap text-white/50 mb-1">TASK DEFINITION</div>
              <div className="body-md text-white font-mono text-xs">{latestResult.execution_summary?.selected_task || "N/A"}</div>
            </div>

            <div className="mb-6">
              <div className="micro-cap text-white/50 mb-1">MODEL PROVENANCE</div>
              <div className="text-xs text-white/80 font-mono bg-white/5 px-2.5 py-1.5 rounded border border-white/10">
                {latestResult.execution_summary?.model_provenance || "SatQuery-RS-Adapted-v1.2 (Fine-tuned)"}
              </div>
            </div>

            <div className="mb-6">
              <div className="micro-cap text-white/50 mb-1">AGENTIC REASONING</div>
              <div className="body-sm text-white/80 leading-relaxed text-xs">{latestResult.execution_summary?.agent_reasoning || "N/A"}</div>
            </div>

            <div className="mb-6">
              <div className="micro-cap text-white/50 mb-2">VISUAL EVIDENCE</div>
              
              {/* GeoJSON Polygon Map Overlay */}
              {latestResult.geojson_data && (
                <Suspense fallback={<div className="h-48 flex items-center justify-center text-white/30 text-xs micro-cap">Loading map...</div>}>
                  <GeoJSONMapOverlay geojsonData={latestResult.geojson_data} />
                </Suspense>
              )}
              
              {/* INTERACTIVE SWIPE SLIDER (IF COMPARISON PAIR EXISTS) */}
              {latestResult.pair_comparison && latestResult.pair_comparison.before_image && latestResult.pair_comparison.after_image && (
                <div className="mb-4">
                  <div className="text-[10px] font-mono tracking-wider uppercase text-cyan-400 font-bold mb-2 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                    <span>INTERACTIVE SUB-PIXEL SWIPE COMPARATOR</span>
                  </div>
                  <SwipeSlider 
                    beforeImage={latestResult.pair_comparison.before_image}
                    afterImage={latestResult.pair_comparison.after_image}
                    beforeLabel={latestResult.pair_comparison.before_label}
                    afterLabel={latestResult.pair_comparison.after_label}
                  />
                </div>
              )}

              {latestResult.visual_evidence && latestResult.visual_evidence.length > 0 ? (
                <div className="flex flex-col gap-4">
                  {latestResult.visual_evidence.map((ev: any, idx: number) => (
                    <div key={idx} className="relative rounded-sm overflow-hidden border border-[#3a3a3f]">
                      {ev.image_base64 ? (
                        <>
                          <div className="absolute top-2 left-2 bg-black/80 px-2 py-1 micro-cap z-10 rounded-sm text-[10px]">
                            {ev.description}
                          </div>
                          <img 
                            src={ev.image_base64} 
                            alt={ev.description} 
                            className="w-full h-auto" 
                          />
                        </>
                      ) : (
                        <pre className="micro-cap text-white/60 p-2 overflow-x-auto">{JSON.stringify(ev, null, 2)}</pre>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="micro-cap text-white/40">NO VISUALS</div>
              )}
            </div>

            {/* DOWNLOAD AUDIT AND GEOJSON FOOTER */}
            <div className="pt-4 border-t border-[#2a2a2f] mt-auto flex flex-col gap-2">
              {latestResult.geojson_data && (
                <button
                  type="button"
                  onClick={handleDownloadGeoJSON}
                  className="w-full py-2.5 px-4 bg-emerald-400 text-black font-semibold text-xs tracking-wider uppercase rounded hover:bg-emerald-300 transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-lg"
                  title="Export Tactical Vector Layers for QGIS / ArcGIS / ISRO Bhuvan"
                >
                  <span>📍</span> EXPORT GEOJSON (.GEOJSON) → QGIS / Bhuvan
                </button>
              )}

              <button
                type="button"
                onClick={handleDownloadReport}
                className="w-full py-2.5 px-4 bg-white text-black font-semibold text-xs tracking-wider uppercase rounded hover:bg-white/80 transition-colors flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>&darr;</span> DOWNLOAD AUDIT REPORT (.PDF)
              </button>
            </div>
            
            </div>
        )}
      </div>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #3a3a3f;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #5a5a5f;
        }
        @keyframes slideUpFade {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slide-up {
          animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
      `}</style>

      {/* Map Explorer Modal */}
      {showMapExplorer && (
        <div className="fixed inset-0 z-[100] bg-black">
          <MapExplorer 
            onCancel={() => setShowMapExplorer(false)}
            onAcquire={async (base64, bbox) => {
              try {
                const res = await fetch(base64);
                const blob = await res.blob();
                const uniqueName = `acquired_satellite_image_${Date.now()}.png`;
                const file = new File([blob], uniqueName, { type: "image/png" });
                setImages(prev => [...prev, file].slice(0, 2));
                setShowMapExplorer(false);
              } catch (e) {
                console.error(e);
              }
            }}
          />
        </div>
      )}

      {/* Lightbox / Modal */}
      {expandedImage && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-md p-4 sm:p-8 animate-in fade-in duration-200 pointer-events-auto"
          onClick={() => setExpandedImage(null)}
        >
          <button 
            className="absolute top-8 right-8 text-white hover:text-[#00F0FF] font-mono tracking-widest uppercase text-sm border border-white/20 px-4 py-2 hover:border-[#00F0FF] transition-colors"
          >
            Close [X]
          </button>
          <div className="relative max-w-5xl w-full max-h-[85vh] flex flex-col items-center justify-center border border-white/10 bg-black shadow-2xl p-2" onClick={(e) => e.stopPropagation()}>
            <img src={expandedImage} alt="Expanded preview" className="w-full h-auto max-h-[80vh] object-contain" />
          </div>
        </div>
      )}

      {/* Mission Presets Modal */}
      <MissionPresetsModal
        isOpen={showMissionPresets}
        onClose={() => setShowMissionPresets(false)}
        onSelectMission={handleSelectMission}
      />
    </main>
  );
}
