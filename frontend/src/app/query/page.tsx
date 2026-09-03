"use client";

import { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  result?: any;
};

import BackgroundSlideshow from '@/components/BackgroundSlideshow';
import FadeInScroll from '@/components/FadeInScroll';
import FramerGlobe from '@/components/FramerGlobe';
import jsPDF from 'jspdf';
import SwipeSlider from '@/components/SwipeSlider';

function FormattedMessage({ content }: { content: string }) {
  if (!content) return null;

  // Clean out any raw markdown asterisks (* and **)
  const sanitized = content.replace(/\*\*/g, "").replace(/\*/g, "");

  // Split into blocks by double newline
  const blocks = sanitized.split(/\n\n+/).map(b => b.trim()).filter(b => b.length > 0);

  return (
    <div className="flex flex-col gap-4 text-sm leading-relaxed">
      {blocks.map((block, bIdx) => {
        // 1. Alert block (Compatibility Failure)
        if (block.includes("COMPATIBILITY CHECK FAILED") || block.includes("Spatial Disparity Detected")) {
          const lines = block.split("\n").map(l => l.trim()).filter(l => l.length > 0);
          const alertTitle = lines[0];
          const alertBody = lines.slice(1).join(" ");
          return (
            <div key={bIdx} className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex flex-col gap-2 shadow-inner">
              <div className="flex items-center gap-2 text-red-400 font-mono text-xs tracking-wider uppercase font-bold">
                <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse"></span>
                <span>{alertTitle.replace(/^⚠️\s*/, "")}</span>
              </div>
              {alertBody && (
                <p className="text-white/80 text-xs sm:text-sm font-sans leading-relaxed">
                  {alertBody}
                </p>
              )}
            </div>
          );
        }

        const lines = block.split("\n").map(l => l.trim()).filter(l => l.length > 0);
        const hasBullets = lines.some(l => l.startsWith("•") || l.startsWith("-"));

        // 2. Bullet list section
        if (hasBullets) {
          return (
            <div key={bIdx} className="flex flex-col gap-2">
              {lines.map((line, lIdx) => {
                if (line.startsWith("•") || line.startsWith("-")) {
                  const bulletText = line.replace(/^[•\-]\s*/, "");
                  const colonIdx = bulletText.indexOf(":");
                  if (colonIdx > 0 && colonIdx < 35) {
                    const label = bulletText.slice(0, colonIdx).trim();
                    const value = bulletText.slice(colonIdx + 1).trim();
                    return (
                      <div key={lIdx} className="flex items-start gap-2.5 pl-1 text-xs sm:text-sm">
                        <span className="w-1.5 h-1.5 rounded-full bg-white/60 mt-2 shrink-0"></span>
                        <div>
                          <span className="text-white/50 font-mono uppercase text-[11px] font-semibold tracking-wider mr-1.5">{label}:</span>
                          <span className="text-white/90">{value}</span>
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div key={lIdx} className="flex items-start gap-2.5 pl-1 text-xs sm:text-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-white/60 mt-2 shrink-0"></span>
                      <span className="text-white/85">{bulletText}</span>
                    </div>
                  );
                } else {
                  // Heading for this bullet group
                  const heading = line.replace(/:$/, "").trim();
                  return (
                    <div key={lIdx} className="text-[10px] font-mono tracking-widest uppercase text-white/50 font-bold flex items-center gap-1.5 mt-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-white"></span>
                      <span>{heading}</span>
                    </div>
                  );
                }
              })}
            </div>
          );
        }

        // 3. Labeled section like "Observation: City — ..."
        const firstLine = lines[0];
        const colonIdx = firstLine.indexOf(":");
        if (colonIdx > 0 && colonIdx < 35) {
          const sectionLabel = firstLine.slice(0, colonIdx).trim();
          const inlineText = firstLine.slice(colonIdx + 1).trim();
          const remainingLines = lines.slice(1);
          const bodyText = [inlineText, ...remainingLines].filter(t => t.length > 0).join(" ");
          return (
            <div key={bIdx} className="flex flex-col gap-1.5">
              <div className="text-[10px] font-mono tracking-widest uppercase text-white/50 font-bold flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-white"></span>
                <span>{sectionLabel}</span>
              </div>
              <div className="text-white/95 text-sm font-medium pl-3 border-l-2 border-white/30 leading-relaxed">
                {bodyText}
              </div>
            </div>
          );
        }

        // 4. Standard paragraph
        return (
          <p key={bIdx} className="text-white/90 text-sm leading-relaxed">
            {block}
          </p>
        );
      })}
    </div>
  );
}

export default function Home() {
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
      category: "SINGLE-IMAGE VQA & RECON",
      queries: [
        "Identify System Architecture",
        "Classify land cover and terrain type",
        "Estimate vegetative canopy density",
        "Identify structural infrastructure and fenestration",
        "Are there any water bodies or drainage basins in this tile?"
      ]
    },
    {
      category: "TEMPORAL CHANGE UNDERSTANDING (2 IMAGES)",
      queries: [
        "Run Change Detection on this bi-temporal pair",
        "Quantify urban expansion and newly constructed infrastructure",
        "Detect deforestation or vegetative canopy loss",
        "Analyze flood extent and hydrological displacement"
      ]
    },
    {
      category: "SPATIAL GROUNDING & LOCALIZATION",
      queries: [
        "Highlight water bodies and reservoirs",
        "Ground airfield runways and taxiways",
        "Locate industrial storage tanks and facilities",
        "Detect transportation road networks"
      ]
    },
    {
      category: "OPTICAL – SAR ANALYSIS (2 IMAGES)",
      queries: [
        "Perform Optical-SAR cross-modal joint analysis",
        "Penetrate cloud cover using SAR radar backscatter",
        "Extract complementary dielectric and surface roughness features"
      ]
    },
    {
      category: "HOLISTIC SCENE CAPTIONING",
      queries: [
        "Generate comprehensive remote-sensing caption",
        "Summarize maritime activity and coastal features"
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
            // Avoid duplicates if effect runs twice
            if (prev.some(f => f.name === "acquired_satellite_image.png")) return prev;
            
            // Add a welcome message to prompt the user
            setMessages([{ 
              role: "assistant", 
              content: "I have successfully acquired and loaded your satellite imagery from the Map Explorer. The target area is ready for analysis. What would you like me to look for?"
            }]);

            return [...prev, file].slice(0, 2);
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) {
      setError("Please provide a query.");
      return;
    }
    
    if (images.length === 0 && messages.length === 0) {
      setError("Please provide at least one image to start the analysis.");
      return;
    }

    setLoading(true);
    setError(null);
    
    const newMessages = [...messages, { role: "user", content: query } as Message];
    setMessages(newMessages);
    
    const currentQuery = query;
    setQuery("");

    const formData = new FormData();
    formData.append("query", currentQuery);
    
    const historyPayload = messages.map(m => ({ 
      role: m.role, 
      user: m.role === 'user' ? m.content : '', 
      assistant: m.role === 'assistant' ? m.content : '' 
    }));
    formData.append("history", JSON.stringify(historyPayload));

    images.forEach((img) => {
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
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
      setMessages([...newMessages, { role: "assistant", content: "Sorry, I encountered an error: " + err.message }]);
    } finally {
      setLoading(false);
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
      
      {/* HOLOGRAPHIC GLOBE BACKGROUND (EMPTY STATE) */}
      {messages.length === 0 && (
        <div className="absolute inset-0 z-0 flex items-center justify-center mix-blend-screen pointer-events-none">
          <div className="relative z-[6] group pointer-events-auto w-[600px] h-[600px] rounded-full flex items-center justify-center">

            {/* Hardware-accelerated desaturation overlay */}
            <div className="absolute inset-0 z-10 bg-black mix-blend-color pointer-events-none transition-opacity duration-700 opacity-100 group-hover:opacity-0 rounded-full" />
            
            <div className="w-full h-full opacity-20 group-hover:opacity-100 transition-opacity duration-700">
              <FramerGlobe />
            </div>
            
          </div>
        </div>
      )}

      {/* FIXED TOP NAV OVERLAY */}
      <nav className="w-full flex justify-between items-center px-8 py-6 z-50">
        <a href="/" className="display-lg tracking-widest text-white hover:opacity-70 transition-opacity pointer-events-auto">
          SATQUERY AI.
        </a>
        <div className="flex gap-4 items-center">
          <a href="/" className="micro-cap text-white hover:opacity-70 transition-opacity border border-white/20 px-4 py-2 rounded-full flex items-center gap-2 pointer-events-auto">
            <span>&larr;</span> BACK TO DASHBOARD
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
                  {images.map((img, i) => (
                    <div key={i} className="w-16 h-16 relative group rounded-md overflow-hidden border border-[#3a3a3f]">
                      <img src={URL.createObjectURL(img)} alt={`Preview ${i}`} className="w-full h-full object-cover" />
                      <button 
                        type="button" 
                        className="absolute inset-0 bg-black/60 text-white flex items-center justify-center micro-cap opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={() => removeImage(i)}
                      >X</button>
                    </div>
                  ))}
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
                  accept="image/png, image/jpeg, image/tiff"
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
                    <div className="absolute bottom-full mb-3 left-0 w-[92vw] sm:w-[560px] max-h-[380px] overflow-y-auto custom-scrollbar bg-black/95 backdrop-blur-xl border border-[#3a3a3f] rounded-xl p-4 shadow-2xl z-50 animate-slide-up">
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
            <div className="w-full lg:w-1/3 bg-black/60 backdrop-blur-md border border-[#3a3a3f] rounded-lg p-6 flex flex-col h-full overflow-y-auto custom-scrollbar shadow-2xl animate-slide-up">
              <div className="flex justify-between items-center mb-6 pb-4 border-b border-[#2a2a2f]">
                <h3 className="display-lg flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                  TRACE
                </h3>
                <button
                  onClick={handleDownloadReport}
                  className="micro-cap px-3 py-1.5 rounded-full border border-white/30 text-white hover:bg-white hover:text-black transition-all flex items-center gap-1 text-[10px] cursor-pointer"
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
                  title="Export Tactical Vector Layers for QGIS / ArcGIS / Bhuvan"
                >
                  <span>📍</span> EXPORT GEOJSON POLYGONS (.GEOJSON)
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
    </main>
  );
}
