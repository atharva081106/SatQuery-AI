"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";

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
  const [maxCC, setMaxCC] = useState(20);
  const [bbox, setBbox] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);

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
    
    // Check if map is already initialized
    const container = mapRef.current;
    if (container._leaflet_id) {
      return;
    }

    // Initialize Map
    const map = L.map(container, {
      center: [20, 0], // Default global view
      zoom: 3,
      zoomControl: false,
    });

    // Dark Matter Base Map (CartoDB) - Perfect for dark space-theme
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }).addTo(map);

    // Add Zoom Control to bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

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
            color: '#00f0ff', // Cyan highlight for the drawn box
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
      
      // Clear previous boxes so we only have one AOI
      drawnItems.clearLayers();
      drawnItems.addLayer(layer);
      
      if (type === 'rectangle') {
        const layerBounds = layer.getBounds();
        // Format: [min_lon, min_lat, max_lon, max_lat]
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
      
      // Convert blob to base64 to store in sessionStorage (since it's a cross-page transition)
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64data = reader.result as string;
        sessionStorage.setItem("satquery_acquired_image", base64data);
        sessionStorage.setItem("satquery_acquired_bbox", JSON.stringify(bbox));
        
        // Navigate to query page
        router.push("/query");
      };
      reader.readAsDataURL(blob);
      
    } catch (e: any) {
      alert(e.message);
      setLoading(false);
    }
  };

  return (
    <div className="w-full h-full relative bg-black">
      <div ref={mapRef} className="absolute inset-0 z-0" />
      
      {/* SpaceX Style Overlay Control Panel */}
      <div className="absolute top-24 left-8 z-10 w-80 pointer-events-none">
        <div className="bg-black/90 border border-[#3a3a3f] p-6 backdrop-blur-md pointer-events-auto shadow-2xl">
          <div className="micro-cap text-[#00f0ff] mb-6 tracking-[2px]">ACQUISITION PARAMETERS</div>
          
          <div className="space-y-6">
            <div>
              <label className="micro-cap block text-white/50 mb-2">TARGET DATASET</label>
              <select 
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full bg-transparent border border-[#3a3a3f] text-white p-2 text-sm focus:outline-none focus:border-white uppercase font-mono"
              >
                <option value="s2">Sentinel-2 (Optical)</option>
                <option value="s1">Sentinel-1 (Radar)</option>
                <option value="l8">Landsat 8</option>
              </select>
            </div>
            
            <div className="flex gap-4">
              <div className="flex-1">
                <label className="micro-cap block text-white/50 mb-2">START DATE</label>
                <input 
                  type="date" 
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full bg-transparent border border-[#3a3a3f] text-white p-2 text-xs focus:outline-none focus:border-white uppercase font-mono" 
                />
              </div>
              <div className="flex-1">
                <label className="micro-cap block text-white/50 mb-2">END DATE</label>
                <input 
                  type="date" 
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full bg-transparent border border-[#3a3a3f] text-white p-2 text-xs focus:outline-none focus:border-white uppercase font-mono" 
                />
              </div>
            </div>

            <div>
              <label className="micro-cap flex justify-between text-white/50 mb-2">
                <span>MAX CLOUD COVER</span>
                <span>{maxCC}%</span>
              </label>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={maxCC}
                onChange={(e) => setMaxCC(parseInt(e.target.value))}
                className="w-full accent-white" 
              />
            </div>

            <button 
              onClick={handleAcquire}
              disabled={loading}
              className={`w-full mt-4 py-3 border border-white text-white transition-colors uppercase tracking-[2px] text-xs font-semibold ${loading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-white hover:text-black cursor-pointer'}`}
            >
              {loading ? 'ACQUIRING...' : 'ACQUIRE DATA'}
            </button>
          </div>
        </div>
      </div>
      
      {/* Target Crosshair */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10 opacity-20">
        <div className="w-8 h-[1px] bg-white absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2" />
        <div className="h-8 w-[1px] bg-white absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2" />
      </div>
    </div>
  );
}
