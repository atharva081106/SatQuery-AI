"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { useAuth } from "@/context/AuthContext";

interface MapExplorerProps {
  onAcquire?: (base64data: string, bbox: number[]) => void;
  onCancel?: () => void;
}

export default function MapExplorer({ onAcquire, onCancel }: MapExplorerProps = {}) {
  const {
    canUseMap,
    incrementMapCount,
    openAuthModal,
    mapCount,
    maxFreeMapQueries,
    isAuthenticated,
  } = useAuth();

  const mapRef = useRef<any>(null);
  const mapInstanceRef = useRef<any>(null);
  const gibsLayerRef = useRef<any>(null);
  const basemapLayerRef = useRef<any>(null);
  const labelLayerRef = useRef<any>(null);
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
  const [configuration, setConfiguration] = useState("Default");
  const [layer, setLayer] = useState("True color");
  const [basemap, setBasemap] = useState<"esri" | "bhuvan" | "osm">("esri");
  const [bbox, setBbox] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [acquisitionMode, setAcquisitionMode] = useState<"single" | "dual">("single");
  const [liveLocationName, setLiveLocationName] = useState<string | null>(null);
  
  const [mousePos, setMousePos] = useState({ lat: 50.16, lng: 20.78 });
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);

  // Sync configuration preset with recommended layer
  useEffect(() => {
    if (configuration === "Agriculture") setLayer("NDVI");
    else if (configuration === "Atmosphere and Air Pollution") setLayer("Highlight Optimized Natural Color");
    else if (configuration === "Floods and Droughts") setLayer("NDWI");
    else if (configuration === "Geology") setLayer("SWIR");
    else if (configuration === "Ocean and Water Bodies") setLayer("NDWI");
    else if (configuration === "Snow and Glaciers") setLayer("NDSI");
    else if (configuration === "Urban") setLayer("False color (urban)");
    else if (configuration === "Default") setLayer("True color");
  }, [configuration]);

  // Adjust layer options if dataset is Radar (s1)
  useEffect(() => {
    if (dataset === "s1") {
      setLayer("True color"); // S1 maps to VV/VH
    }
  }, [dataset]);

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
      attributionControl: false,
    });
    mapInstanceRef.current = map;

    // Default basemap: ESRI World Imagery
    basemapLayerRef.current = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 18
      }
    ).addTo(map);

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
        // Reopen config panel on mobile so user can immediately click Acquire Data
        if (typeof window !== "undefined" && window.innerWidth < 640) {
          setMobilePanelOpen(true);
        }
      }
    });

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [L]);

  // Fetch location name when bbox changes
  useEffect(() => {
    if (!bbox) {
      setLiveLocationName(null);
      return;
    }
    
    let isMounted = true;
    const fetchLocation = async () => {
      try {
        setLiveLocationName("Locating...");
        const centerLat = (bbox[1] + bbox[3]) / 2;
        const centerLng = (bbox[0] + bbox[2]) / 2;
        const res = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${centerLat}&lon=${centerLng}&format=json`);
        if (res.ok && isMounted) {
          const data = await res.json();
          setLiveLocationName(data.display_name || "Unknown Location");
        }
      } catch (e) {
        if (isMounted) setLiveLocationName("Unknown Location");
      }
    };
    fetchLocation();
    
    return () => { isMounted = false; };
  }, [bbox]);

  // Switch basemap when selection changes
  useEffect(() => {
    if (!L || !mapInstanceRef.current) return;
    if (basemapLayerRef.current) {
      mapInstanceRef.current.removeLayer(basemapLayerRef.current);
    }
    if (labelLayerRef.current) {
      mapInstanceRef.current.removeLayer(labelLayerRef.current);
    }
    const BASEMAPS: Record<string, any> = {
      esri: L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        { attribution: 'Tiles &copy; Esri', maxZoom: 18 }
      ),
      bhuvan: L.tileLayer(
        'https://bhuvan-vec2.nrsc.gov.in/bhuvan/gwc/service/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=india_map&STYLE=default&TILEMATRIXSET=EPSG:900913&TILEMATRIX=EPSG:900913:{z}&TILEROW={y}&TILECOL={x}&FORMAT=image/png',
        {
          attribution: 'ISRO Bhuvan &copy; NRSC',
          maxZoom: 14,
          tileSize: 256,
          errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
        }
      ),
      osm: L.tileLayer(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }
      ),
    };
    const newLayer = BASEMAPS[basemap];
    newLayer.addTo(mapInstanceRef.current);
    newLayer.bringToBack();
    basemapLayerRef.current = newLayer;

    // Add labels overlay if not OSM (OSM already has labels)
    if (basemap !== 'osm') {
      labelLayerRef.current = L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { maxZoom: 20 }
      ).addTo(mapInstanceRef.current);
    }
  }, [basemap, L]);

  // Dynamic NASA GIBS Tile Layer Effect
  useEffect(() => {
    if (!L || !mapInstanceRef.current) return;
    
    // Remove old layer if it exists
    if (gibsLayerRef.current) {
      mapInstanceRef.current.removeLayer(gibsLayerRef.current);
    }
    
    // Create new layer with the updated endDate
    const gibsLayer = L.tileLayer(
      `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/${endDate}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
      {
        attribution: 'NASA GIBS',
        maxZoom: 9,
        opacity: 0.9,
        className: 'gibs-layer',
        bounds: [[-85.0511287776, -180], [85.0511287776, 180]]
      }
    );
    
    gibsLayer.addTo(mapInstanceRef.current);
    gibsLayerRef.current = gibsLayer;
    
  }, [L, endDate]);

  const handleAcquire = async () => {
    if (!canUseMap) {
      openAuthModal("You have used your 1 free Map Satellite Acquisition. Please sign in or create an account to unlock unlimited satellite area tasking.");
      return;
    }

    if (!bbox) {
      alert("Please draw an Area of Interest (Bounding Box) on the map first.");
      return;
    }
    
    setLoading(true);
    
    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      
      // Step 1: Use pre-fetched location if available, otherwise fetch
      let locationName = liveLocationName && liveLocationName !== "Locating..." ? liveLocationName : "Unknown Location";
      if (locationName === "Unknown Location") {
        try {
          const centerLat = (bbox[1] + bbox[3]) / 2;
          const centerLng = (bbox[0] + bbox[2]) / 2;
          const geoRes = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${centerLat}&lon=${centerLng}&format=json`);
          if (geoRes.ok) {
            const geoData = await geoRes.json();
            locationName = geoData.display_name || "Unknown Location";
          }
        } catch (e) {
          console.warn("Geocoding failed", e);
        }
      }
      
      const fetchImage = async (targetDate: string) => {
        // Compute start_date (1 day prior) to strictly enforce the selected date rather than pulling an old image
        const target = new Date(targetDate);
        target.setDate(target.getDate() - 1);
        const searchStartDate = target.toISOString().split('T')[0];

        const res = await fetch(`${backendUrl}/api/acquire`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            bbox,
            start_date: searchStartDate,
            end_date: targetDate,
            dataset,
            maxcc: maxCC,
            configuration,
            layer
          })
        });
        
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || `Failed to acquire imagery for ${targetDate}`);
        }
        
        const blob = await res.blob();
        return new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      };

      if (acquisitionMode === "single") {
        const base64data = await fetchImage(endDate);
        incrementMapCount();
        if (onAcquire) {
          onAcquire(base64data, bbox);
        } else {
          sessionStorage.setItem("satquery_acquired_image", base64data);
          sessionStorage.setItem("satquery_acquired_bbox", JSON.stringify(bbox));
          sessionStorage.setItem("satquery_acquired_layer", layer);
          sessionStorage.setItem("satquery_location_name", locationName);
          sessionStorage.setItem("satquery_target_date", endDate);
          sessionStorage.setItem("satquery_acquisition_mode", "single");
          router.push("/query");
        }
      } else {
        // Dual Mode
        const [base64_1, base64_2] = await Promise.all([
          fetchImage(startDate),
          fetchImage(endDate)
        ]);
        
        incrementMapCount();
        if (onAcquire) {
          // If onAcquire is provided, it doesn't currently support multiple images, so just pass the first one.
          onAcquire(base64_1, bbox);
        } else {
          sessionStorage.setItem("satquery_acquired_image_1", base64_1);
          sessionStorage.setItem("satquery_acquired_image_2", base64_2);
          sessionStorage.setItem("satquery_acquired_bbox", JSON.stringify(bbox));
          sessionStorage.setItem("satquery_acquired_layer", layer);
          sessionStorage.setItem("satquery_location_name", locationName);
          sessionStorage.setItem("satquery_target_date_1", startDate);
          sessionStorage.setItem("satquery_target_date_2", endDate);
          sessionStorage.setItem("satquery_acquisition_mode", "dual");
          router.push("/query");
        }
      }
      
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
      
      {/* FLOATING ACQUISITION PANEL (Collapsible on Mobile, Persistent on Desktop) */}
      <div 
        data-lenis-prevent
        className={`fixed sm:absolute inset-x-3 bottom-20 sm:bottom-auto sm:top-18 sm:left-6 w-[calc(100vw-1.5rem)] sm:w-[350px] max-h-[85dvh] sm:max-h-[calc(100vh-5.5rem)] overflow-y-auto custom-scrollbar bg-black/90 backdrop-blur-2xl border border-white/20 p-3 sm:p-4 z-[450] flex flex-col gap-2.5 shadow-2xl rounded-2xl sm:rounded-xl pointer-events-auto touch-pan-y animate-slide-up ${
          mobilePanelOpen ? 'flex' : 'hidden sm:flex'
        }`}
        onWheel={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-white/20 pb-1.5 shrink-0">
          <h2 className="text-xs font-bold tracking-[0.2em] uppercase font-mono">Acquisition Config</h2>
          <div className="flex items-center gap-2">
            {isAuthenticated ? (
              <span className="text-[9px] font-bold text-white bg-white/10 border border-white/30 px-2 py-0.5 rounded-full uppercase tracking-wider">
                PRO UNLIMITED
              </span>
            ) : (
              <span className="text-[9px] font-bold text-[#00F0FF] bg-[#00F0FF]/10 border border-[#00F0FF]/30 px-2 py-0.5 rounded-full uppercase tracking-wider">
                {mapCount >= maxFreeMapQueries ? "0/1 FREE" : "1/1 FREE"}
              </span>
            )}
            <button
              type="button"
              onClick={() => setMobilePanelOpen(false)}
              className="sm:hidden text-white/50 hover:text-white text-xs px-2 py-0.5 border border-white/20 rounded cursor-pointer"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Basemap Switcher */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-white/50 tracking-widest uppercase font-mono">Base Map</label>
          <div className="grid grid-cols-3 gap-1">
            {([['esri', 'ESRI Sat'], ['bhuvan', '🇮🇳 Bhuvan'], ['osm', 'OpenStreet']] as const).map(([val, label]) => (
              <button
                key={val}
                type="button"
                onClick={() => setBasemap(val)}
                className={`text-[9px] sm:text-[10px] py-1 px-1 uppercase tracking-wider border transition-all ${
                  basemap === val
                    ? 'border-[#00F0FF] text-[#00F0FF] bg-[#00F0FF]/10 font-bold'
                    : 'border-white/20 text-white/50 hover:border-white/50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {basemap === 'bhuvan' && (
            <div className="text-[9px] text-[#00F0FF]/80 tracking-wider font-mono">
              ✓ ISRO Bhuvan NRSC — National Geoportal
            </div>
          )}
        </div>
        
        {/* Dataset & Configuration (2-Column Compact Row) */}
        <div className="grid grid-cols-2 gap-2">
          {/* Dataset Selection */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-white/50 tracking-widest uppercase font-mono truncate">Target Dataset</label>
            <select 
              value={dataset} 
              onChange={e => setDataset(e.target.value)}
              className="bg-black/60 border border-white/20 text-white text-[11px] px-2 py-1.5 outline-none focus:border-[#00F0FF] uppercase tracking-wider truncate"
            >
              <option value="s2" className="bg-black">Sentinel-2 (Optical)</option>
              <option value="s1" className="bg-black">Sentinel-1 (Radar)</option>
              <option value="l8" className="bg-black">Landsat 8-9</option>
            </select>
          </div>

          {/* Configuration */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-white/50 tracking-widest uppercase font-mono truncate">Configuration</label>
            <select 
              value={configuration} 
              onChange={e => setConfiguration(e.target.value)}
              className="bg-black/60 border border-white/20 text-white text-[11px] px-2 py-1.5 outline-none focus:border-[#00F0FF] uppercase tracking-wider truncate"
            >
              <option value="Default" className="bg-black">Default</option>
              <option value="Monitoring Earth from Space" className="bg-black">Monitoring Earth</option>
              <option value="Agriculture" className="bg-black">Agriculture</option>
              <option value="Atmosphere and Air Pollution" className="bg-black">Atmosphere & Air</option>
              <option value="Change Detection through Time" className="bg-black">Change Detection</option>
              <option value="Floods and Droughts" className="bg-black">Floods & Droughts</option>
              <option value="Geology" className="bg-black">Geology</option>
              <option value="Ocean and Water Bodies" className="bg-black">Ocean & Water</option>
              <option value="Snow and Glaciers" className="bg-black">Snow & Glaciers</option>
              <option value="Urban" className="bg-black">Urban</option>
            </select>
          </div>
        </div>

        {/* Spectral Layer */}
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <label className="text-[10px] text-white/50 tracking-widest uppercase font-mono">Data Layer</label>
            {configuration !== "Default" ? (
              <span className="text-[8px] text-[#00F0FF] tracking-widest uppercase animate-pulse">RECOMMENDED</span>
            ) : (
              <span className="text-[8px] text-white/40 italic">Preview: True Color</span>
            )}
          </div>
          <select 
            value={layer} 
            onChange={e => setLayer(e.target.value)}
            className="bg-black/60 border border-white/20 text-white text-[11px] px-2.5 py-1.5 outline-none focus:border-[#00F0FF] uppercase tracking-wider"
          >
            <option value="True color" className="bg-black">{dataset === 's1' ? 'Radar VV/VH' : 'True color'}</option>
            {dataset !== "s1" && (
              <>
                <option value="False color" className="bg-black">False color (NIR)</option>
                <option value="Highlight Optimized Natural Color" className="bg-black">Highlight Optimized Natural Color</option>
                <option value="Wildfires" className="bg-black">Wildfires</option>
                <option value="NDVI" className="bg-black">NDVI (Vegetation)</option>
                <option value="False color (urban)" className="bg-black">False color (Urban)</option>
                <option value="Moisture index" className="bg-black">Moisture index (NDMI)</option>
                <option value="SWIR" className="bg-black">SWIR</option>
                <option value="NDWI" className="bg-black">NDWI (Water)</option>
                <option value="NDSI" className="bg-black">NDSI (Snow)</option>
                <option value="Scene classification map" className="bg-black">Scene classification map</option>
              </>
            )}
          </select>
        </div>

        {/* Acquisition Mode & Date Range */}
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <label className="text-[10px] text-white/50 tracking-widest uppercase font-mono">
              {acquisitionMode === "single" ? "Target Date" : "Date Range"}
            </label>
            <div className="inline-flex rounded border border-white/20 p-0.5 bg-black/40">
              <button
                type="button"
                onClick={() => setAcquisitionMode("single")}
                className={`text-[9px] px-2 py-0.5 uppercase tracking-wider transition-all rounded-[2px] ${
                  acquisitionMode === "single"
                    ? 'bg-[#00F0FF]/20 text-[#00F0FF] font-bold'
                    : 'text-white/50 hover:text-white'
                }`}
              >
                Single
              </button>
              <button
                type="button"
                onClick={() => setAcquisitionMode("dual")}
                className={`text-[9px] px-2 py-0.5 uppercase tracking-wider transition-all rounded-[2px] ${
                  acquisitionMode === "dual"
                    ? 'bg-[#00F0FF]/20 text-[#00F0FF] font-bold'
                    : 'text-white/50 hover:text-white'
                }`}
              >
                Change Detect
              </button>
            </div>
          </div>

          {acquisitionMode === "single" ? (
            <DatePicker
              selected={endDate ? new Date(endDate) : null}
              onChange={(date: Date | null) => {
                if (date) setEndDate(date.toISOString().split('T')[0]);
              }}
              dateFormat="dd/MM/yy"
              className="w-full bg-black/60 border border-white/20 text-white text-[11px] px-2.5 py-1.5 outline-none focus:border-[#00F0FF]"
            />
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-[8px] text-white/40 uppercase tracking-wider block mb-0.5">Before (T1)</span>
                <DatePicker
                  selected={startDate ? new Date(startDate) : null}
                  onChange={(date: Date | null) => {
                    if (date) setStartDate(date.toISOString().split('T')[0]);
                  }}
                  dateFormat="dd/MM/yy"
                  className="w-full bg-black/60 border border-white/20 text-white text-[11px] px-2 py-1.5 outline-none focus:border-[#00F0FF]"
                />
              </div>
              <div>
                <span className="text-[8px] text-white/40 uppercase tracking-wider block mb-0.5">After (T2)</span>
                <DatePicker
                  selected={endDate ? new Date(endDate) : null}
                  onChange={(date: Date | null) => {
                    if (date) setEndDate(date.toISOString().split('T')[0]);
                  }}
                  dateFormat="dd/MM/yy"
                  className="w-full bg-black/60 border border-white/20 text-white text-[11px] px-2 py-1.5 outline-none focus:border-[#00F0FF]"
                />
              </div>
            </div>
          )}
        </div>

        {/* Cloud Cover */}
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center text-[10px]">
            <label className="text-white/50 tracking-widest uppercase font-mono">Max Cloud Cover</label>
            <span className="font-bold text-[#00F0FF] font-mono">{maxCC}%</span>
          </div>
          <input 
            type="range" 
            min="0" max="100" 
            value={maxCC} 
            onChange={e => setMaxCC(parseInt(e.target.value))} 
            className="w-full h-1.5 accent-[#00F0FF] bg-white/10 rounded cursor-pointer"
          />
        </div>

        {/* Action Button */}
        <button 
          type="button"
          onClick={handleAcquire}
          disabled={loading}
          className={`w-full py-2.5 mt-1 text-[11px] font-bold tracking-[0.2em] uppercase transition-all duration-300 border shrink-0 cursor-pointer ${
            loading 
            ? 'border-white/20 text-white/40 cursor-not-allowed bg-transparent' 
            : 'border-white text-white hover:bg-white hover:text-black hover:shadow-[0_0_15px_rgba(255,255,255,0.5)]'
          }`}
        >
          {loading ? 'Initiating Scan...' : 'Acquire Data'}
        </button>
        
        {onCancel && (
          <button 
            type="button"
            onClick={onCancel}
            disabled={loading}
            className="w-full py-1.5 text-[10px] font-bold tracking-[0.2em] uppercase transition-all duration-300 border border-white/20 text-white/60 hover:text-white hover:border-white shrink-0 cursor-pointer"
          >
            Cancel
          </button>
        )}
        
        {/* Helper text */}
        {!bbox && (
          <div className="text-[9px] text-[#00F0FF] tracking-widest uppercase text-center animate-pulse">
            Draw bounding box to enable
          </div>
        )}
      </div>

      {/* LAT/LNG TRACKER & LOCATION */}
      <div className="absolute top-16 sm:bottom-6 right-3 sm:right-16 z-[400] pointer-events-none flex flex-col items-end gap-2">
        {liveLocationName && (
          <div className="bg-black/80 backdrop-blur-md border border-[#00F0FF]/30 px-3 py-2 text-[10px] tracking-widest text-[#00F0FF] rounded-lg max-w-[250px] text-right truncate">
            {liveLocationName}
          </div>
        )}
        <div className="bg-black/70 backdrop-blur-md border border-white/10 px-2.5 sm:px-4 py-1.5 sm:py-2 text-[9px] sm:text-[10px] tracking-[0.15em] sm:tracking-[0.2em] text-[#00F0FF] rounded-lg">
          LAT: {mousePos.lat.toFixed(3)} / LNG: {mousePos.lng.toFixed(3)}
        </div>
      </div>

      {/* MOBILE FLOATING TOGGLE PILL */}
      <div className="sm:hidden absolute bottom-8 left-1/2 -translate-x-1/2 z-[480] pointer-events-auto pb-[env(safe-area-inset-bottom)]">
        <button
          type="button"
          onClick={() => setMobilePanelOpen(!mobilePanelOpen)}
          className={`px-5 py-2.5 rounded-full font-mono text-xs font-bold tracking-widest shadow-2xl flex items-center gap-2 uppercase cursor-pointer transition-all ${
            mobilePanelOpen
              ? "bg-white text-black border border-white"
              : "bg-black/90 text-white border border-[#00F0FF]/60 shadow-[0_0_15px_rgba(0,240,255,0.4)]"
          }`}
        >
          <span>{mobilePanelOpen ? "🗺️ VIEW FULL MAP" : "⚙️ CONFIGURE SCAN"}</span>
          {bbox && <span className="w-2 h-2 rounded-full bg-[#00F0FF] animate-pulse" />}
        </button>
      </div>
      
    </div>
  );
}
