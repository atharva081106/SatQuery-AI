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
            color: '#8bc34a', // Light green Copernicus style highlight
            weight: 2,
            fillOpacity: 0.2
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
    <div className="w-full h-full relative bg-gray-100 flex font-sans overflow-hidden">
      
      {/* LEFT SIDEBAR (Copernicus Style) */}
      <div className="w-[380px] h-full bg-[#f8f9fa] flex flex-col z-50 shadow-2xl shrink-0">
        
        {/* Blue Header Area */}
        <div className="bg-[#003399] text-white flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-blue-800">
            <div className="flex items-center gap-2">
              <span className="font-bold text-xl tracking-tight">Copernicus</span>
              <span className="text-sm font-light text-blue-200 mt-1">BROWSER</span>
            </div>
            <div className="flex items-center gap-4 text-xs font-semibold">
              <div className="flex items-center gap-1 cursor-pointer">EN <ChevronDown size={14}/></div>
              <div className="flex items-center gap-1 cursor-pointer">User Account <ChevronDown size={14}/></div>
              <div className="bg-blue-800 p-1 rounded cursor-pointer" onClick={() => router.push('/')} title="Back to Dashboard">
                <ChevronLeft size={16}/>
              </div>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="flex w-full">
            <button 
              onClick={() => setActiveTab("VISUALISE")}
              className={`flex-1 py-3 text-sm font-bold tracking-wider ${activeTab === 'VISUALISE' ? 'bg-[#002266] text-[#8bc34a] border-b-2 border-[#8bc34a]' : 'text-white hover:bg-blue-800'}`}
            >
              VISUALISE
            </button>
            <button 
              onClick={() => setActiveTab("SEARCH")}
              className={`flex-1 py-3 text-sm font-bold tracking-wider ${activeTab === 'SEARCH' ? 'bg-[#002266] text-[#8bc34a] border-b-2 border-[#8bc34a]' : 'text-white hover:bg-blue-800'}`}
            >
              SEARCH
            </button>
          </div>
        </div>

        {/* Dashboard / Workspace Links */}
        <div className="flex bg-white border-b border-gray-200 text-xs font-semibold text-blue-700">
          <div className="flex-1 py-2 text-center border-r border-gray-200 hover:bg-blue-50 cursor-pointer flex justify-center items-center gap-2">
            <LayoutPanelTop size={14}/> DASHBOARD
          </div>
          <div className="flex-1 py-2 text-center hover:bg-blue-50 cursor-pointer flex justify-center items-center gap-2">
            <FileDown size={14}/> WORKSPACE
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 flex flex-col gap-4">
          
          {/* DATE ACCORDION */}
          <div className="bg-white border border-gray-200 shadow-sm rounded-sm overflow-hidden">
            <div className="flex justify-between items-center p-3 bg-gray-50 border-b border-gray-200">
              <span className="font-bold text-[#003399] text-sm tracking-wide">DATE: RANGE</span>
              <div className="flex gap-1">
                <div className="w-6 h-6 rounded bg-[#003399] flex items-center justify-center text-white"><Calendar size={14}/></div>
                <div className="w-6 h-6 rounded border border-gray-300 flex items-center justify-center text-gray-500"><Globe size={14}/></div>
              </div>
            </div>
            <div className="p-4 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="flex-1 border border-gray-300 rounded px-2 py-1.5 text-sm font-semibold text-gray-700" />
                <span className="text-gray-400 font-bold">-</span>
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="flex-1 border border-gray-300 rounded px-2 py-1.5 text-sm font-semibold text-gray-700" />
              </div>
              <div className="flex items-center gap-3">
                <Cloud size={18} className="text-gray-400"/>
                <input type="range" min="0" max="100" value={maxCC} onChange={e => setMaxCC(parseInt(e.target.value))} className="flex-1" />
                <span className="text-sm font-bold text-gray-500 w-8">{maxCC}%</span>
              </div>
              <button 
                onClick={handleAcquire}
                disabled={loading}
                className={`w-full py-2.5 mt-2 rounded flex items-center justify-center gap-2 text-white font-bold text-sm shadow transition-colors ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#a3d166] hover:bg-[#8bc34a] cursor-pointer'}`}
              >
                {loading ? 'ACQUIRING DATA...' : 'Fetch Satellite Imagery'} <span className="text-lg leading-none">↗</span>
              </button>
            </div>
          </div>

          {/* CONFIGURATION ACCORDION */}
          <div className="bg-white border border-gray-200 shadow-sm rounded-sm overflow-hidden">
            <div className="flex justify-between items-center p-3 bg-gray-50">
              <span className="font-bold text-[#003399] text-sm tracking-wide">CONFIGURATION:</span>
            </div>
            <div className="px-4 pb-4">
              <select className="w-full border-b border-gray-300 py-2 text-sm font-semibold text-gray-700 focus:outline-none focus:border-[#003399] appearance-none cursor-pointer">
                <option>Default True Color</option>
                <option>False Color (Vegetation)</option>
                <option>NDVI</option>
              </select>
            </div>
          </div>

          {/* DATA COLLECTIONS ACCORDION */}
          <div className="bg-white border border-gray-200 shadow-sm rounded-sm overflow-hidden">
            <div className="flex justify-between items-center p-3 bg-gray-50 border-b border-gray-200">
              <span className="font-bold text-[#003399] text-sm tracking-wide">DATA COLLECTIONS:</span>
              <div className="w-6 h-6 rounded bg-[#003399] flex items-center justify-center text-white"><Layers size={14}/></div>
            </div>
            
            <div className="flex flex-col">
              {/* Sentinel 2 */}
              <div className="border-b border-gray-100">
                <div className="px-4 py-2 flex justify-between items-center cursor-pointer hover:bg-gray-50">
                  <span className="font-bold text-sm text-gray-800">Sentinel-2</span>
                  <ChevronDown size={16} className="text-gray-500"/>
                </div>
                <div className="bg-gray-50 px-4 py-1 flex flex-col gap-1 pb-2">
                  <div className="flex justify-between items-center py-1.5 px-2 rounded hover:bg-gray-200 cursor-pointer text-sm text-gray-700">
                    <span>Sentinel-2 L1C</span>
                    <Info size={14} className="text-gray-400"/>
                  </div>
                  <div 
                    onClick={() => setDataset("s2")}
                    className={`flex justify-between items-center py-1.5 px-2 rounded cursor-pointer text-sm font-semibold ${dataset === 's2' ? 'bg-gray-200 text-black' : 'hover:bg-gray-200 text-gray-700'}`}
                  >
                    <span>Sentinel-2 L2A</span>
                    {dataset === 's2' ? <Check size={16} className="text-[#003399]"/> : <Info size={14} className="text-gray-400"/>}
                  </div>
                </div>
              </div>

              {/* Sentinel 1 */}
              <div className="border-b border-gray-100">
                <div className="px-4 py-2 flex justify-between items-center cursor-pointer hover:bg-gray-50" onClick={() => setDataset("s1")}>
                  <span className={`font-bold text-sm ${dataset === 's1' ? 'text-[#003399]' : 'text-gray-800'}`}>Sentinel-1</span>
                  {dataset === 's1' ? <Check size={16} className="text-[#003399]"/> : <ChevronDown size={16} className="text-gray-500"/>}
                </div>
              </div>

              {/* Landsat */}
              <div>
                <div className="px-4 py-2 flex justify-between items-center cursor-pointer hover:bg-gray-50" onClick={() => setDataset("l8")}>
                  <span className={`font-bold text-sm ${dataset === 'l8' ? 'text-[#003399]' : 'text-gray-800'}`}>Landsat 8-9</span>
                  {dataset === 'l8' ? <Check size={16} className="text-[#003399]"/> : <ChevronDown size={16} className="text-gray-500"/>}
                </div>
              </div>

            </div>
          </div>

        </div>

        {/* Footer Logos */}
        <div className="bg-[#003399] h-12 shrink-0 flex items-center justify-between px-4 text-white text-xs">
          <div className="flex gap-4">
            <span className="font-bold border border-white px-2 rounded-sm text-[10px]">EU</span>
            <span className="font-bold tracking-tight">Copernicus</span>
            <span className="font-bold tracking-tight">eesa</span>
          </div>
          <div className="flex gap-3 text-[10px] font-semibold underline">
            <a href="#">About</a>
            <a href="#">Support</a>
          </div>
        </div>

      </div>

      {/* MAP AREA */}
      <div className="flex-1 relative z-0">
        
        {/* Yellow Banner */}
        <div className="absolute top-0 left-0 w-full flex justify-center mt-4 z-[400] pointer-events-none">
          <div className="bg-[#ffcc00] text-black font-bold px-6 py-2 shadow-md">
            Please draw a bounding box using the toolbar on the right
          </div>
        </div>

        <div ref={mapRef} className="absolute inset-0 z-0" />
        
        {/* RIGHT TOOLBARS OVERLAYS */}
        <div className="absolute top-4 right-4 z-[400] flex gap-2 pointer-events-none">
          
          {/* Search Bar */}
          <div className="bg-white rounded shadow-sm h-10 w-64 flex items-center px-3 border border-gray-200 pointer-events-auto">
            <Search size={16} className="text-[#003399] mr-2"/>
            <input type="text" placeholder="Go to Place" className="flex-1 outline-none text-sm font-semibold text-gray-700"/>
          </div>

          {/* Toolbar Group 1 */}
          <div className="flex flex-col gap-2 pointer-events-auto">
            <button className="w-10 h-10 bg-white border border-gray-200 shadow-sm flex items-center justify-center text-[#003399] hover:bg-gray-50"><Layers size={20}/></button>
            <button className="w-10 h-10 bg-white border border-gray-200 shadow-sm flex items-center justify-center text-[#003399] hover:bg-gray-50"><Info size={20}/></button>
          </div>
          
        </div>

        <div className="absolute top-20 right-4 z-[400] flex flex-col gap-2 pointer-events-auto">
          {/* Toolbar Group 2 */}
          <div className="flex flex-col border border-gray-200 shadow-sm bg-white rounded overflow-hidden">
            <button className="w-10 h-10 border-b border-gray-100 flex items-center justify-center text-[#003399] hover:bg-gray-50"><Upload size={18}/></button>
            <button className="w-10 h-10 border-b border-gray-100 flex items-center justify-center text-[#003399] hover:bg-gray-50"><Pencil size={18}/></button>
            <button className="w-10 h-10 flex items-center justify-center text-[#8bc34a] hover:bg-gray-50"><Ruler size={18}/></button>
          </div>

          <div className="flex flex-col border border-gray-200 shadow-sm bg-white rounded overflow-hidden mt-4">
            <button className="w-10 h-10 border-b border-gray-100 flex items-center justify-center text-[#003399] hover:bg-gray-50"><ImageIcon size={18}/></button>
            <button className="w-10 h-10 border-b border-gray-100 flex items-center justify-center text-[#003399] hover:bg-gray-50"><Film size={18}/></button>
            <button className="w-10 h-10 border-b border-gray-100 flex items-center justify-center text-[#003399] font-bold text-sm hover:bg-gray-50">3D</button>
            <button className="w-10 h-10 flex items-center justify-center text-[#003399] hover:bg-gray-50"><BarChart3 size={18}/></button>
          </div>
        </div>

        {/* BOTTOM RIGHT OVERLAYS */}
        <div className="absolute bottom-0 right-0 z-[400] flex flex-col items-end gap-2 p-1 pointer-events-none">
          <div className="bg-white/80 backdrop-blur text-[10px] font-mono px-2 py-0.5 border border-gray-300 text-gray-700 pointer-events-auto mr-12">
            Lat: {mousePos.lat.toFixed(2)}, Lng: {mousePos.lng.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
