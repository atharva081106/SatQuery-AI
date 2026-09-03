"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import { 
  Search, Layers, Info, Upload, Edit2, Pencil, Ruler, Download, 
  Image as ImageIcon, Film, Box, BarChart3, ChevronLeft, Calendar, 
  Globe, Cloud, ChevronDown, Check, LayoutPanelTop, FileDown
} from "lucide-react";

// We must dynamically import leaflet to avoid SSR issues with window
export default function MapExplorer() {
  const mapRef = useRef<any>(null);
  const router = useRouter();
  const [L, setL] = useState<any>(null);
  
  // UI State
  const [dataset, setDataset] = useState("s2");
  
  // Default dates (last 30 days)
  const today = new Date();
  const thirtyDaysAgo = new Date(today);
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  
  const [startDate, setStartDate] = useState(thirtyDaysAgo.toISOString().split('T')[0]);
  const [endDate, setEndDate] = useState(today.toISOString().split('T')[0]);
  const [maxCC, setMaxCC] = useState(30);
  const [bbox, setBbox] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  
  const [activeTab, setActiveTab] = useState("VISUALISE");
  const [mousePos, setMousePos] = useState({ lat: 50.16, lng: 20.78 });

  useEffect(() => {
    // Dynamic import of Leaflet
    const initLeaflet = async () => {
      const leaflet = (await import("leaflet")).default;
      require("leaflet-draw");
      setL(leaflet);
    };
    initLeaflet();
  }, []);

  useEffect(() => {
    if (!L || !mapRef.current) return;
    
    const container = mapRef.current;
    if (container._leaflet_id) {
      return;
    }

    // Initialize Map on Europe
    const map = L.map(container, {
      center: [50.16, 20.78],
      zoom: 5,
      zoomControl: false,
    });

    // High Res Satellite Imagery (Esri World Imagery)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
      maxZoom: 18
    }).addTo(map);

    // Add Zoom Control to bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Track mouse position for the bottom right coordinate display
    map.on('mousemove', (e: any) => {
      setMousePos({ lat: e.latlng.lat, lng: e.latlng.lng });
    });

    // Initialize FeatureGroup to store editable layers
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    // Add Draw Control
    const drawControl = new L.Control.Draw({
      position: 'topright',
      edit: {
        featureGroup: drawnItems
      },
      draw: {
        polygon: false,
        polyline: false,
        circle: false,
        marker: false,
        circlemarker: false,
        rectangle: {
          shapeOptions: {
            color: '#00F0FF', // Neon Cyan for SpaceX styling
            weight: 2,
            fillOpacity: 0.1
          }
        }
      }
    });
    map.addControl(drawControl);

    // Listen to draw events
    map.on(L.Draw.Event.CREATED, (e: any) => {
      const type = e.layerType;
      const layer = e.layer;
      
      drawnItems.clearLayers();
      drawnItems.addLayer(layer);
      
      if (type === 'rectangle') {
        const layerBounds = layer.getBounds();
        setBbox([
          layerBounds.getWest(),
          layerBounds.getSouth(),
          layerBounds.getEast(),
          layerBounds.getNorth()
        ]);
      }
    });

    return () => {
      map.remove();
    };
  }, [L]);

  const handleAcquire = async () => {
    if (!bbox) {
      alert("Please draw an Area of Interest (Bounding Box) on the map first.");
      return;
    }
    
    setLoading(true);
    
    try {
      const res = await fetch("http://127.0.0.1:8000/api/acquire", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          bbox,
          start_date: startDate,
          end_date: endDate,
          dataset,
          maxcc: maxCC
        })
      });
      
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to acquire imagery");
      }
      
      const blob = await res.blob();
      
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64data = reader.result as string;
        sessionStorage.setItem("satquery_acquired_image", base64data);
        sessionStorage.setItem("satquery_acquired_bbox", JSON.stringify(bbox));
        
        router.push("/query");
      };
      reader.readAsDataURL(blob);
      
    } catch (e: any) {
      alert(e.message);
      setLoading(false);
    }
  };

  return (
    <div className="w-full h-full relative bg-black flex font-mono overflow-hidden text-white">
      
      {/* LEFT SIDEBAR (SpaceX/Glassmorphism Style) */}
      <div className="w-[380px] h-full bg-black/60 backdrop-blur-xl border-r border-white/10 flex flex-col z-[400] shrink-0">
        
        {/* Header Area */}
        <div className="flex flex-col border-b border-white/10">
          <div className="flex items-center justify-between px-6 py-5">
            <div className="flex items-center gap-2">
              <span className="font-bold text-2xl tracking-[0.2em]">SATQUERY AI.</span>
            </div>
            <div className="flex items-center gap-4 text-xs font-semibold tracking-widest text-white/50">
              <div className="border border-white/20 p-2 hover:bg-white hover:text-black hover:border-white transition-colors cursor-pointer" onClick={() => router.push('/')} title="ABORT">
                <ChevronLeft size={16}/>
              </div>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="flex w-full border-t border-white/10">
            <button 
              onClick={() => setActiveTab("VISUALISE")}
              className={`flex-1 py-4 text-xs font-bold tracking-[0.15em] transition-colors ${activeTab === 'VISUALISE' ? 'bg-white/10 text-[#00F0FF] border-b-2 border-[#00F0FF]' : 'text-white hover:bg-white/5'}`}
            >
              VISUALISE
            </button>
            <button 
              onClick={() => setActiveTab("SEARCH")}
              className={`flex-1 py-4 text-xs font-bold tracking-[0.15em] transition-colors ${activeTab === 'SEARCH' ? 'bg-white/10 text-[#00F0FF] border-b-2 border-[#00F0FF]' : 'text-white hover:bg-white/5'}`}
            >
              SEARCH
            </button>
          </div>
        </div>

        {/* Dashboard / Workspace Links */}
        <div className="flex bg-black/40 border-b border-white/10 text-[10px] font-bold tracking-[0.15em] text-white/60">
          <div className="flex-1 py-3 text-center border-r border-white/10 hover:bg-white/10 hover:text-white transition-colors cursor-pointer flex justify-center items-center gap-2">
            <LayoutPanelTop size={14}/> DASHBOARD
          </div>
          <div className="flex-1 py-3 text-center hover:bg-white/10 hover:text-white transition-colors cursor-pointer flex justify-center items-center gap-2">
            <FileDown size={14}/> WORKSPACE
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col gap-6">
          
          {/* DATE ACCORDION */}
          <div className="bg-black/40 border border-white/10 backdrop-blur-sm overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b border-white/10 bg-white/5">
              <span className="font-bold text-xs tracking-[0.15em]">DATE: RANGE</span>
              <div className="flex gap-2">
                <div className="w-7 h-7 flex items-center justify-center text-[#00F0FF]"><Calendar size={16}/></div>
              </div>
            </div>
            <div className="p-5 flex flex-col gap-5">
              <div className="flex items-center gap-3">
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="flex-1 bg-transparent border border-white/20 px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-[#00F0FF] transition-colors" />
                <span className="text-white/40 font-bold">-</span>
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="flex-1 bg-transparent border border-white/20 px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-[#00F0FF] transition-colors" />
              </div>
              <div className="flex items-center gap-4">
                <Cloud size={18} className="text-white/40"/>
                <input type="range" min="0" max="100" value={maxCC} onChange={e => setMaxCC(parseInt(e.target.value))} className="flex-1 accent-[#00F0FF]" />
                <span className="text-xs font-bold text-white/60 w-8">{maxCC}%</span>
              </div>
              <button 
                onClick={handleAcquire}
                disabled={loading}
                className={`w-full py-4 mt-2 border border-white/20 tracking-[0.2em] text-xs font-bold transition-colors uppercase ${loading ? 'bg-white/10 text-white/40 cursor-not-allowed' : 'bg-transparent text-white hover:bg-white hover:text-black cursor-pointer'}`}
              >
                {loading ? 'ACQUIRING...' : 'FETCH IMAGERY'}
              </button>
            </div>
          </div>

          {/* CONFIGURATION ACCORDION */}
          <div className="bg-black/40 border border-white/10 backdrop-blur-sm overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b border-white/10 bg-white/5">
              <span className="font-bold text-xs tracking-[0.15em]">CONFIGURATION</span>
            </div>
            <div className="p-4">
              <select className="w-full bg-transparent border-b border-white/20 py-2 text-xs font-mono text-white/80 focus:outline-none focus:border-[#00F0FF] appearance-none cursor-pointer">
                <option className="bg-black">DEFAULT TRUE COLOR</option>
                <option className="bg-black">FALSE COLOR (VEGETATION)</option>
                <option className="bg-black">NDVI</option>
              </select>
            </div>
          </div>

          {/* DATA COLLECTIONS ACCORDION */}
          <div className="bg-black/40 border border-white/10 backdrop-blur-sm overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b border-white/10 bg-white/5">
              <span className="font-bold text-xs tracking-[0.15em]">DATA COLLECTIONS</span>
              <div className="text-[#00F0FF]"><Layers size={16}/></div>
            </div>
            
            <div className="flex flex-col">
              {/* Sentinel 2 */}
              <div className="border-b border-white/10">
                <div className="px-5 py-4 flex justify-between items-center cursor-pointer hover:bg-white/5 transition-colors">
                  <span className="font-bold text-xs tracking-wider">SENTINEL-2</span>
                  <ChevronDown size={16} className="text-white/40"/>
                </div>
                <div className="bg-black/20 px-5 py-2 flex flex-col gap-2 pb-4">
                  <div className="flex justify-between items-center py-2 px-3 hover:bg-white/10 transition-colors cursor-pointer text-xs text-white/60">
                    <span className="tracking-widest">L1C</span>
                    <Info size={14} className="text-white/30"/>
                  </div>
                  <div 
                    onClick={() => setDataset("s2")}
                    className={`flex justify-between items-center py-2 px-3 transition-colors cursor-pointer text-xs font-bold ${dataset === 's2' ? 'bg-white/10 text-[#00F0FF] border border-white/10' : 'hover:bg-white/10 text-white/80'}`}
                  >
                    <span className="tracking-widest">L2A</span>
                    {dataset === 's2' ? <Check size={16} className="text-[#00F0FF]"/> : <Info size={14} className="text-white/30"/>}
                  </div>
                </div>
              </div>

              {/* Sentinel 1 */}
              <div className="border-b border-white/10">
                <div className="px-5 py-4 flex justify-between items-center cursor-pointer hover:bg-white/5 transition-colors" onClick={() => setDataset("s1")}>
                  <span className={`font-bold text-xs tracking-wider ${dataset === 's1' ? 'text-[#00F0FF]' : 'text-white'}`}>SENTINEL-1</span>
                  {dataset === 's1' ? <Check size={16} className="text-[#00F0FF]"/> : <ChevronDown size={16} className="text-white/40"/>}
                </div>
              </div>

              {/* Landsat */}
              <div>
                <div className="px-5 py-4 flex justify-between items-center cursor-pointer hover:bg-white/5 transition-colors" onClick={() => setDataset("l8")}>
                  <span className={`font-bold text-xs tracking-wider ${dataset === 'l8' ? 'text-[#00F0FF]' : 'text-white'}`}>LANDSAT 8-9</span>
                  {dataset === 'l8' ? <Check size={16} className="text-[#00F0FF]"/> : <ChevronDown size={16} className="text-white/40"/>}
                </div>
              </div>

            </div>
          </div>

        </div>

      </div>

      {/* MAP AREA */}
      <div className="flex-1 relative z-0">
        
        {/* Yellow/Cyan Banner */}
        <div className="absolute top-0 left-0 w-full flex justify-center mt-6 z-[400] pointer-events-none">
          <div className="bg-black/80 backdrop-blur-md border border-[#00F0FF]/30 text-[#00F0FF] font-bold text-xs tracking-[0.2em] px-8 py-3 uppercase">
            Define Target Area Using The Toolbar
          </div>
        </div>

        <div ref={mapRef} className="absolute inset-0 z-0" />
        
        {/* RIGHT TOOLBARS OVERLAYS (Glassmorphism) */}
        <div className="absolute top-6 right-6 z-[400] flex gap-3 pointer-events-none">
          
          {/* Search Bar */}
          <div className="bg-black/60 backdrop-blur-md border border-white/10 h-12 w-72 flex items-center px-4 pointer-events-auto hover:bg-black/80 transition-colors">
            <Search size={18} className="text-white/50 mr-3"/>
            <input type="text" placeholder="LOCATE TARGET..." className="flex-1 bg-transparent outline-none text-xs font-bold tracking-[0.15em] text-white placeholder-white/30"/>
          </div>

          {/* Toolbar Group 1 */}
          <div className="flex flex-col gap-2 pointer-events-auto">
            <button className="w-12 h-12 bg-black/60 backdrop-blur-md border border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><Layers size={20}/></button>
            <button className="w-12 h-12 bg-black/60 backdrop-blur-md border border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><Info size={20}/></button>
          </div>
          
        </div>

        <div className="absolute top-24 right-6 z-[400] flex flex-col gap-4 pointer-events-auto mt-2">
          {/* Toolbar Group 2 */}
          <div className="flex flex-col border border-white/10 bg-black/60 backdrop-blur-md">
            <button className="w-12 h-12 border-b border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><Upload size={18}/></button>
            <button className="w-12 h-12 border-b border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><Pencil size={18}/></button>
            <button className="w-12 h-12 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><Ruler size={18}/></button>
          </div>

          <div className="flex flex-col border border-white/10 bg-black/60 backdrop-blur-md">
            <button className="w-12 h-12 border-b border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><ImageIcon size={18}/></button>
            <button className="w-12 h-12 border-b border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><Film size={18}/></button>
            <button className="w-12 h-12 border-b border-white/10 flex items-center justify-center text-white hover:bg-white hover:text-black font-bold text-xs tracking-wider transition-colors">3D</button>
            <button className="w-12 h-12 flex items-center justify-center text-white hover:bg-white hover:text-black transition-colors"><BarChart3 size={18}/></button>
          </div>
        </div>

        {/* BOTTOM RIGHT OVERLAYS */}
        <div className="absolute bottom-6 right-20 z-[400] flex flex-col items-end gap-2 p-1 pointer-events-none">
          <div className="bg-black/60 backdrop-blur-md text-[10px] tracking-[0.2em] font-mono px-4 py-2 border border-white/10 text-[#00F0FF] pointer-events-auto">
            LAT: {mousePos.lat.toFixed(4)} / LNG: {mousePos.lng.toFixed(4)}
          </div>
        </div>
      </div>
    </div>
  );
}
