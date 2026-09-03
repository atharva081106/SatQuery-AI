"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";

interface MapExplorerProps {
  onAcquire?: (base64data: string, bbox: number[]) => void;
  onCancel?: () => void;
}

export default function MapExplorer({ onAcquire, onCancel }: MapExplorerProps = {}) {
  const mapRef = useRef<any>(null);
  const router = useRouter();
  const [L, setL] = useState<any>(null);
  
  // UI State
  const [dataset, setDataset] = useState("s2");
  const today = new Date();
  const thirtyDaysAgo = new Date(today);
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  
  const [startDate, setStartDate] = useState(thirtyDaysAgo.toISOString().split('T')[0]);
  const [endDate, setEndDate] = useState(today.toISOString().split('T')[0]);
  const [maxCC, setMaxCC] = useState(20);
  const [bbox, setBbox] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  
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

    const map = L.map(container, {
      center: [50.16, 20.78],
      zoom: 5,
      zoomControl: false,
    });

    // High Res Satellite Imagery (Esri World Imagery)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP',
      maxZoom: 18
    }).addTo(map);

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    map.on('mousemove', (e: any) => {
      setMousePos({ lat: e.latlng.lat, lng: e.latlng.lng });
    });

    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

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
            color: '#00F0FF',
            weight: 2,
            fillOpacity: 0.1
          }
        }
      }
    });
    map.addControl(drawControl);

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
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${backendUrl}/api/acquire`, {
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
        if (onAcquire) {
          onAcquire(base64data, bbox);
        } else {
          sessionStorage.setItem("satquery_acquired_image", base64data);
          sessionStorage.setItem("satquery_acquired_bbox", JSON.stringify(bbox));
          router.push("/query");
        }
      };
      reader.readAsDataURL(blob);
      
    } catch (e: any) {
      alert(e.message);
      setLoading(false);
    }
  };

  return (
    <div className="w-full h-full relative font-mono text-white bg-black">
      
      {/* CSS to invert Leaflet native controls to match dark mode perfectly */}
      <style dangerouslySetInnerHTML={{__html: `
        .leaflet-control-container .leaflet-bar {
          filter: invert(1) hue-rotate(180deg);
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3) !important;
        }
        input[type="date"]::-webkit-calendar-picker-indicator {
          filter: invert(1);
        }
      `}} />

      {/* THE MAP */}
      <div ref={mapRef} className="absolute inset-0 z-0" />
      
      {/* FLOATING ACQUISITION PANEL (design.md spec) */}
      <div className="absolute top-28 left-8 w-80 bg-black/60 backdrop-blur-md border border-white/20 p-6 z-[400] flex flex-col gap-6 shadow-2xl">
        <h2 className="text-sm font-bold tracking-[0.2em] uppercase border-b border-white/20 pb-2">Acquisition Config</h2>
        
        {/* Dataset Selection */}
        <div className="flex flex-col gap-2">
          <label className="text-xs text-white/50 tracking-widest uppercase">Target Dataset</label>
          <select 
            value={dataset} 
            onChange={e => setDataset(e.target.value)}
            className="bg-transparent border border-white/20 text-white text-xs px-3 py-2 outline-none focus:border-[#00F0FF] uppercase tracking-wider"
          >
            <option value="s2" className="bg-black">Sentinel-2 (Optical)</option>
            <option value="s1" className="bg-black">Sentinel-1 (Radar)</option>
            <option value="l8" className="bg-black">Landsat 8-9</option>
          </select>
        </div>

        {/* Date Range */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between">
            <label className="text-xs text-white/50 tracking-widest uppercase">Start Date</label>
            <label className="text-xs text-white/50 tracking-widest uppercase">End Date</label>
          </div>
          <div className="flex items-center gap-2">
            <input 
              type="date" 
              value={startDate} 
              onChange={e => setStartDate(e.target.value)} 
              className="w-full bg-transparent border border-white/20 text-white text-xs px-2 py-2 outline-none focus:border-[#00F0FF] [color-scheme:dark]"
            />
            <input 
              type="date" 
              value={endDate} 
              onChange={e => setEndDate(e.target.value)} 
              className="w-full bg-transparent border border-white/20 text-white text-xs px-2 py-2 outline-none focus:border-[#00F0FF] [color-scheme:dark]"
            />
          </div>
        </div>

        {/* Cloud Cover */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <label className="text-xs text-white/50 tracking-widest uppercase">Max Cloud Cover</label>
            <span className="text-xs font-bold text-[#00F0FF]">{maxCC}%</span>
          </div>
          <input 
            type="range" 
            min="0" max="100" 
            value={maxCC} 
            onChange={e => setMaxCC(parseInt(e.target.value))} 
            className="w-full accent-[#00F0FF]"
          />
        </div>

        {/* Action Button */}
        <button 
          onClick={handleAcquire}
          disabled={loading}
          className={`w-full py-3 mt-4 text-xs font-bold tracking-[0.2em] uppercase transition-all duration-300 border ${
            loading 
            ? 'border-white/20 text-white/40 cursor-not-allowed bg-transparent' 
            : 'border-white text-white hover:bg-white hover:text-black hover:shadow-[0_0_15px_rgba(255,255,255,0.5)]'
          }`}
        >
          {loading ? 'Initiating Scan...' : 'Acquire Data'}
        </button>
        
        {onCancel && (
          <button 
            onClick={onCancel}
            disabled={loading}
            className="w-full py-3 mt-2 text-xs font-bold tracking-[0.2em] uppercase transition-all duration-300 border border-white/20 text-white/60 hover:text-white hover:border-white"
          >
            Cancel
          </button>
        )}
        
        {/* Helper text */}
        {!bbox && (
          <div className="text-[10px] text-[#00F0FF] tracking-widest uppercase text-center mt-2 animate-pulse">
            Draw bounding box to enable
          </div>
        )}
      </div>

      {/* LAT/LNG TRACKER */}
      <div className="absolute bottom-6 right-16 z-[400] pointer-events-none">
        <div className="bg-black/60 backdrop-blur-md border border-white/10 px-4 py-2 text-[10px] tracking-[0.2em] text-[#00F0FF]">
          LAT: {mousePos.lat.toFixed(4)} / LNG: {mousePos.lng.toFixed(4)}
        </div>
      </div>
      
    </div>
  );
}
