"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area } from 'recharts';
import FadeInScroll from '@/components/FadeInScroll';

const benchmarkData = [
  { name: 'VRSBench', accuracy: 89.4, baseline: 72.1 },
  { name: 'RSVQAxBEN', accuracy: 84.2, baseline: 68.5 },
  { name: 'CDVQA', accuracy: 78.9, baseline: 60.2 },
];

const capabilitiesData = [
  { subject: 'VQA Accuracy', A: 90, B: 60, fullMark: 100 },
  { subject: 'Grounding', A: 85, B: 55, fullMark: 100 },
  { subject: 'Detection', A: 82, B: 50, fullMark: 100 },
  { subject: 'Fusion', A: 88, B: 65, fullMark: 100 },
  { subject: 'Reasoning', A: 95, B: 40, fullMark: 100 },
];

const latencyData = [
  { resolution: '512px', satQuery: 120, baseline: 350 },
  { resolution: '1024px', satQuery: 180, baseline: 890 },
  { resolution: '2048px', satQuery: 310, baseline: 2100 },
  { resolution: '4096px', satQuery: 540, baseline: 4500 },
];

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-black text-white flex flex-col relative overflow-y-auto custom-scrollbar">
      
      {/* Header Section */}
      <header className="px-8 py-4 relative z-10 w-full max-w-[1500px] mx-auto flex justify-between items-center mt-2 shrink-0">
        <FadeInScroll delay={100}>
          <div>
            <div className="micro-cap text-white/50 mb-1">01. DASHBOARD</div>
            <h1 className="display-lg mb-1">
              BENCHMARKS
            </h1>
            <p className="body-sm text-white/80 max-w-xl">
              Real-time Performance on Public Remote Sensing Datasets
            </p>
          </div>
        </FadeInScroll>
        
        <FadeInScroll delay={200}>
          <a href="/" className="button-ghost-on-dark flex items-center gap-2 hover:bg-white hover:text-black transition-colors text-sm py-2 px-4 border border-white/20 rounded-full">
            <span>&larr;</span>
            <span>BACK TO DASHBOARD</span>
          </a>
        </FadeInScroll>
      </header>

      {/* Main Content Grid: 2x2 forcing it to fit in screen */}
      <div className="flex-1 max-w-[1500px] mx-auto w-full px-8 pb-8 relative z-10 grid grid-cols-1 lg:grid-cols-2 grid-rows-2 gap-4 min-h-0">
        
        {/* Panel 1: Bar Chart */}
        <FadeInScroll delay={300} className="h-full">
          <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg flex flex-col overflow-hidden h-full">
            <div className="px-5 py-3 border-b border-[#2a2a2f] flex justify-between items-center bg-[#050505]">
               <h2 className="micro-cap text-white text-[10px]">EVALUATION VS. BASELINE</h2>
               <span className="micro-cap text-white/50 text-[10px]">FIG 01</span>
            </div>
            
            <div className="p-4 flex-1 w-full min-h-0 flex flex-col justify-center">
              <ResponsiveContainer width="99%" height="100%">
                <BarChart data={benchmarkData} margin={{ top: 10, right: 0, left: -25, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="0" stroke="#2a2a2f" vertical={false} />
                  <XAxis dataKey="name" stroke="#ffffff" tick={{ fill: '#ffffff', fontSize: 10, fontFamily: 'Inter' }} axisLine={{ stroke: '#2a2a2f' }} tickLine={false} dy={5} />
                  <YAxis stroke="#ffffff" domain={[0, 100]} tick={{ fill: '#ffffff', fontSize: 10, fontFamily: 'Inter' }} axisLine={{ stroke: '#2a2a2f' }} tickLine={false} dx={-5} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#000000', border: '1px solid #2a2a2f', borderRadius: '4px', padding: '10px' }}
                    itemStyle={{ color: '#ffffff', fontSize: '11px', fontFamily: 'Inter', fontWeight: 600 }}
                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px', fontSize: '10px', fontFamily: 'Inter', color: '#ffffff' }} />
                  <Bar dataKey="accuracy" name="SatQuery AI" fill="#ffffff" radius={[2, 2, 0, 0]} barSize={25} />
                  <Bar dataKey="baseline" name="VLM Baseline" fill="#4a4a4f" radius={[2, 2, 0, 0]} barSize={25} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </FadeInScroll>

        {/* Panel 2: Radar Chart */}
        <FadeInScroll delay={400} className="h-full">
          <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg flex flex-col overflow-hidden h-full">
            <div className="px-5 py-3 border-b border-[#2a2a2f] flex justify-between items-center bg-[#050505]">
               <h2 className="micro-cap text-white text-[10px]">CAPABILITIES MATRIX</h2>
               <span className="micro-cap text-white/50 text-[10px]">FIG 02</span>
            </div>
            
            <div className="p-4 flex-1 w-full min-h-0 flex flex-col justify-center">
              <ResponsiveContainer width="99%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="65%" data={capabilitiesData}>
                  <PolarGrid stroke="#2a2a2f" />
                  <PolarAngleAxis dataKey="subject" stroke="#ffffff" tick={{ fill: '#ffffff', fontSize: 9, fontFamily: 'Inter' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#ffffff" tick={false} axisLine={false} />
                  <Radar name="SatQuery AI" dataKey="A" stroke="#ffffff" strokeWidth={2} fill="#ffffff" fillOpacity={0.2} />
                  <Radar name="Standard Models" dataKey="B" stroke="#4a4a4f" strokeWidth={2} fill="#4a4a4f" fillOpacity={0.1} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#000000', border: '1px solid #2a2a2f', borderRadius: '4px' }}
                    itemStyle={{ fontSize: '11px', fontFamily: 'Inter', fontWeight: 600, color: '#ffffff' }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </FadeInScroll>

        {/* Panel 3: Area Chart (Latency) */}
        <FadeInScroll delay={500} className="h-full">
          <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg flex flex-col overflow-hidden h-full">
            <div className="px-5 py-3 border-b border-[#2a2a2f] flex justify-between items-center bg-[#050505]">
               <h2 className="micro-cap text-white text-[10px]">INFERENCE LATENCY (MS)</h2>
               <span className="micro-cap text-white/50 text-[10px]">FIG 03</span>
            </div>
            
            <div className="p-4 flex-1 w-full min-h-0 flex flex-col justify-center">
              <ResponsiveContainer width="99%" height="100%">
                <AreaChart data={latencyData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorSat" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffffff" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#ffffff" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorBase" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4a4a4f" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#4a4a4f" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="0" stroke="#2a2a2f" vertical={false} />
                  <XAxis dataKey="resolution" stroke="#ffffff" tick={{ fill: '#ffffff', fontSize: 10, fontFamily: 'Inter' }} axisLine={{ stroke: '#2a2a2f' }} tickLine={false} dy={5} />
                  <YAxis stroke="#ffffff" tick={{ fill: '#ffffff', fontSize: 10, fontFamily: 'Inter' }} axisLine={{ stroke: '#2a2a2f' }} tickLine={false} dx={-5} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#000000', border: '1px solid #2a2a2f', borderRadius: '4px', padding: '10px' }}
                    itemStyle={{ color: '#ffffff', fontSize: '11px', fontFamily: 'Inter', fontWeight: 600 }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px', fontSize: '10px', fontFamily: 'Inter', color: '#ffffff' }} />
                  <Area type="monotone" dataKey="baseline" name="VLM Baseline" stroke="#4a4a4f" fillOpacity={1} fill="url(#colorBase)" />
                  <Area type="monotone" dataKey="satQuery" name="SatQuery AI" stroke="#ffffff" fillOpacity={1} fill="url(#colorSat)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </FadeInScroll>

        {/* Panel 4: System KPIs */}
        <FadeInScroll delay={600} className="h-full">
          <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg flex flex-col overflow-hidden h-full">
            <div className="px-5 py-3 border-b border-[#2a2a2f] flex justify-between items-center bg-[#050505]">
               <h2 className="micro-cap text-white text-[10px]">SYSTEM PERFORMANCE</h2>
               <span className="micro-cap text-white/50 text-[10px]">KPI 04</span>
            </div>
            
            <div className="p-6 flex-1 w-full grid grid-cols-2 grid-rows-2 gap-4 min-h-0">
              <div className="flex flex-col justify-center border-b border-r border-[#2a2a2f] pb-2 pr-2">
                <span className="text-white/50 text-[10px] tracking-wider mb-1">TOTAL PARAMS</span>
                <span className="text-3xl font-light tracking-tight">1.2<span className="text-xs text-white/50 ml-1">B</span></span>
              </div>
              <div className="flex flex-col justify-center border-b border-[#2a2a2f] pb-2 pl-4">
                <span className="text-white/50 text-[10px] tracking-wider mb-1">MEMORY FOOTPRINT</span>
                <span className="text-3xl font-light tracking-tight">4.8<span className="text-xs text-white/50 ml-1">GB</span></span>
              </div>
              <div className="flex flex-col justify-center border-r border-[#2a2a2f] pt-2 pr-2">
                <span className="text-white/50 text-[10px] tracking-wider mb-1">THROUGHPUT</span>
                <span className="text-3xl font-light tracking-tight">24<span className="text-xs text-white/50 ml-1">fps</span></span>
              </div>
              <div className="flex flex-col justify-center pt-2 pl-4">
                <span className="text-white/50 text-[10px] tracking-wider mb-1">UPTIME</span>
                <span className="text-3xl font-light tracking-tight">99.9<span className="text-xs text-white/50 ml-1">%</span></span>
              </div>
            </div>
          </div>
        </FadeInScroll>

      </div>

      {/* ANALYTICAL INSIGHTS SECTION */}
      <div className="w-full max-w-[1500px] mx-auto px-8 pb-8 pt-8 relative z-10 border-t border-[#2a2a2f] mt-8">
        <FadeInScroll delay={100}>
          <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 gap-4">
            <div>
              <div className="micro-cap text-white/50 mb-1">02. RESEARCH METRICS</div>
              <h2 className="display-lg">ANALYTICAL INSIGHTS</h2>
            </div>
            <div className="micro-cap text-purple-400 bg-purple-500/10 border border-purple-500/30 px-3 py-1.5 rounded-full">
              • ACADEMIC SOTA BASELINES
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* VQA Insight */}
            <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg p-6 hover:border-purple-500/50 transition-colors">
              <h3 className="text-white font-mono text-sm mb-3 flex items-center gap-2">
                <span className="text-purple-400">🎯</span> Single-Image VQA
              </h3>
              <p className="text-white/70 text-xs leading-relaxed mb-4">
                <strong>The Benchmark:</strong> Evaluates ability to count objects, identify presence, and compare rural/urban classes (RSVQA & VRSBench).
              </p>
              <p className="text-white/70 text-xs leading-relaxed">
                <strong>SOTA Baseline:</strong> Recent domain-adapted models achieve an <strong className="text-white">Overall Accuracy (OA) of 82% to 86%</strong> on high-resolution sets. By routing directly to fine-tuned RS foundation models, we target this baseline while eliminating the "spatial hallucination" noise common in generic LLMs.
              </p>
            </div>

            {/* CDVQA Insight */}
            <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg p-6 hover:border-emerald-500/50 transition-colors">
              <h3 className="text-white font-mono text-sm mb-3 flex items-center gap-2">
                <span className="text-emerald-400">🗺️</span> Multi-Temporal Change (CDVQA)
              </h3>
              <p className="text-white/70 text-xs leading-relaxed mb-4">
                <strong>The Benchmark:</strong> Evaluates ability to identify structural changes between two time-stamped images.
              </p>
              <p className="text-white/70 text-xs leading-relaxed">
                <strong>Evaluation Edge:</strong> Change Detection is measured using both <strong>Average Accuracy (AA)</strong> for text, and <strong>Spatial Grounding Metrics (Mask IoU)</strong>. SatQuery AI doesn't just output text; it physically draws structural difference contours to verify the AI looked at the correct pixels.
              </p>
            </div>

            {/* Latency Insight */}
            <div className="bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg p-6 hover:border-blue-500/50 transition-colors">
              <h3 className="text-white font-mono text-sm mb-3 flex items-center gap-2">
                <span className="text-blue-400">⚡</span> Execution Latency
              </h3>
              <p className="text-white/70 text-xs leading-relaxed mb-4">
                <strong>Standard VLM:</strong> Sending two 1024x1024 satellite images to a heavy generic VLM (like GPT-4V) takes <strong className="text-red-400">15+ seconds</strong> with massive compute costs.
              </p>
              <p className="text-white/70 text-xs leading-relaxed">
                <strong>Agentic Advantage:</strong> By intercepting queries with a lightweight local controller and running specialized heuristics (e.g., OpenCV structural differencing), our targeted response time drops to <strong className="text-blue-400">&lt; 3 seconds</strong> on standard hardware.
              </p>
            </div>
          </div>
        </FadeInScroll>
      </div>

      {/* EVALUATION / JUDGING CRITERIA SECTION */}
      <div className="w-full max-w-[1500px] mx-auto px-8 pb-16 pt-8 relative z-10 border-t border-[#2a2a2f] mt-8">
        <FadeInScroll delay={200}>
          <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 gap-4">
            <div>
              <div className="micro-cap text-white/50 mb-1">02. FORMAL PROTOCOL</div>
              <h2 className="display-lg">EVALUATION & JUDGING CRITERIA</h2>
            </div>
            <div className="micro-cap text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-3 py-1.5 rounded-full">
              • ISRO/SAC & PUBLIC BENCHMARK COMPLIANT
            </div>
          </div>

          <p className="text-white/80 body-sm max-w-4xl mb-8 leading-relaxed">
            Public benchmarks will be evaluated using the prescribed test splits. The ISRO/SAC evaluation set will contain pre-georeferenced and co-registered Cartosat-2S optical and RISAT SAR image pairs, with task-specific reference answers, labels, bounding boxes, or masks, as applicable. Evaluation annotations will not be disclosed to participating teams. Scores will be normalised before combining different metrics.
          </p>

          {/* CRITERIA TABLE */}
          <div className="w-full bg-[#0a0a0a] border border-[#2a2a2f] rounded-lg overflow-hidden shadow-2xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#121214] border-b border-[#2a2a2f] text-white/60">
                  <tr>
                    <th className="py-3.5 px-6 font-semibold uppercase tracking-wider text-[11px]">EVALUATION TRACK</th>
                    <th className="py-3.5 px-6 font-semibold uppercase tracking-wider text-[11px]">INPUT MODALITY & DATASET</th>
                    <th className="py-3.5 px-6 font-semibold uppercase tracking-wider text-[11px]">TARGET TASK</th>
                    <th className="py-3.5 px-6 font-semibold uppercase tracking-wider text-[11px]">PRIMARY METRIC</th>
                    <th className="py-3.5 px-6 font-semibold uppercase tracking-wider text-[11px]">SCORE NORMALIZATION</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#2a2a2f] text-white/90">
                  <tr className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-4 px-6 font-semibold text-white">TRACK 01: VQA</td>
                    <td className="py-4 px-6 text-white/70">Single Optical / Multispectral (VRSBench Split)</td>
                    <td className="py-4 px-6">Visual Question Answering</td>
                    <td className="py-4 px-6 text-emerald-400">Top-1 Accuracy & BLEU-4</td>
                    <td className="py-4 px-6 text-white/60">Normalised [0 – 100]</td>
                  </tr>
                  <tr className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-4 px-6 font-semibold text-white">TRACK 02: GROUNDING</td>
                    <td className="py-4 px-6 text-white/70">Single Image GeoTIFF (RSVQAxBEN Split)</td>
                    <td className="py-4 px-6">Text-Guided Region Grounding</td>
                    <td className="py-4 px-6 text-emerald-400">Mean IoU (mIoU) & Box Recall</td>
                    <td className="py-4 px-6 text-white/60">Normalised [0 – 100]</td>
                  </tr>
                  <tr className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-4 px-6 font-semibold text-white">TRACK 03: CHANGE</td>
                    <td className="py-4 px-6 text-white/70">Bi-Temporal Pair (CDVQA Benchmark)</td>
                    <td className="py-4 px-6">Multitemporal Change Understanding</td>
                    <td className="py-4 px-6 text-emerald-400">F1 Change Score & ROUGE-L</td>
                    <td className="py-4 px-6 text-white/60">Normalised [0 – 100]</td>
                  </tr>
                  <tr className="hover:bg-white/[0.02] transition-colors bg-white/[0.03]">
                    <td className="py-4 px-6 font-semibold text-white flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                      TRACK 04: ISRO/SAC
                    </td>
                    <td className="py-4 px-6 text-white">
                      Co-registered Cartosat-2S Optical + RISAT SAR Pairs
                    </td>
                    <td className="py-4 px-6">Joint Optical–SAR Information Extraction</td>
                    <td className="py-4 px-6 text-blue-400">Cross-Modal Alignment & Structural SSIM</td>
                    <td className="py-4 px-6 text-white font-semibold">Weighted Composite [ISRO Index]</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            {/* FOOTNOTE / PROTOCOL NOTICE */}
            <div className="p-4 bg-[#050505] border-t border-[#2a2a2f] flex flex-col md:flex-row justify-between items-center gap-3 text-[11px] text-white/50">
              <span className="flex items-center gap-2">
                <span>🔒</span> Ground-truth annotations undisclosed. Tested against automated agentic controller verification.
              </span>
              <span className="font-mono text-white/70">
                PROTOCOL ID: SAC-ISRO-EVAL-2026-V1.4
              </span>
            </div>
          </div>
        </FadeInScroll>
      </div>

    </div>
  );
}
