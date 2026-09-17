"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { useAuth } from "@/context/AuthContext";
import { 
  Search, 
  RotateCw, 
  RotateCcw, 
  Navigation, 
  X, 
  MapPin, 
  Crosshair, 
  Loader2,
  Dices,
  SlidersHorizontal,
  Compass
} from "lucide-react";

interface MapExplorerProps {
  onAcquire?: (base64data: string, bbox: number[]) => void;
  onCancel?: () => void;
}

// Curated Earth Observation and ISRO strategic hubs
const TARGET_PRESETS = [
  { name: "Mumbai Naval Dockyard", location: "Mumbai, Maharashtra, India", lat: 18.9288, lon: 72.8447, desc: "Naval port & coastal shipping" },
  { name: "Sriharikota (SDSC SHAR)", location: "Sriharikota, Andhra Pradesh, India", lat: 13.7199, lon: 80.2305, desc: "ISRO primary orbital launch center" },
  { name: "New Delhi Capital", location: "Delhi, India", lat: 28.6139, lon: 77.2090, desc: "Urban infrastructure & administration" },
  { name: "Bengaluru ISRO HQ", location: "Bengaluru, Karnataka, India", lat: 12.9716, lon: 77.5946, desc: "Space agency command headquarters" },
  { name: "Brahmaputra Flood Basin", location: "Guwahati, Assam, India", lat: 26.1445, lon: 91.7362, desc: "Dynamic riverine flood ecosystem" },
  { name: "Sundarbans Delta", location: "West Bengal, India", lat: 21.9497, lon: 89.1833, desc: "World's largest mangrove reserve" },
  { name: "Suez Canal Transit", location: "Ismailia, Egypt", lat: 30.5852, lon: 32.2654, desc: "Strategic maritime waterway" },
  { name: "Tokyo Bay Coastal Zone", location: "Tokyo, Japan", lat: 35.6762, lon: 139.6503, desc: "Reclaimed coastal megacity infrastructure" },
  { name: "Cape Canaveral Space Force", location: "Florida, United States", lat: 28.3922, lon: -80.6077, desc: "NASA & commercial space launch pads" },
];

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

  // Map Rotation & Bearing State (Arbitrary & Random Degrees)
  const [bearing, setBearing] = useState(0);
  const [isDraggingCompass, setIsDraggingCompass] = useState(false);
  const [showRotationFineTune, setShowRotationFineTune] = useState(false);
  const compassDiscRef = useRef<HTMLDivElement>(null);
  const rotationFineTuneRef = useRef<HTMLDivElement>(null);

  // Search & Auto-Zoom State
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const searchMarkerRef = useRef<any>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);

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
    // Dynamic import of Leaflet and plugins
    const initLeaflet = async () => {
      const leaflet = (await import("leaflet")).default;
      if (typeof window !== "undefined") {
        (window as any).L = leaflet;
        try {
          require("leaflet-rotate");
        } catch (e) {
          console.warn("[MapExplorer] leaflet-rotate error:", e);
        }
        try {
          require("leaflet-draw");
        } catch (e) {
          console.warn("[MapExplorer] leaflet-draw error:", e);
        }
      }
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

    const map = (L as any).map(container, {
      center: [50.16, 20.78],
      zoom: 5,
      zoomControl: false,
      attributionControl: false,
      rotate: true,
      bearing: 0,
      touchRotate: true,
      shiftKeyRotate: true,
      rotateControl: false,
    });
    mapInstanceRef.current = map;

    // Listen for rotation events from touch gesture or programmatic rotation
    if (typeof map.on === "function") {
      map.on("rotate", () => {
        const b = typeof map.getBearing === "function" ? Math.round(map.getBearing()) : 0;
        setBearing(((b % 360) + 360) % 360);
      });
    }

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

  // Click outside to close search dropdown & rotation fine-tune popover
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(e.target as Node)) {
        setShowSearchDropdown(false);
      }
      if (rotationFineTuneRef.current && !rotationFineTuneRef.current.contains(e.target as Node)) {
        setShowRotationFineTune(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Search Pin Management: remove pin marker from map when user clears search
  const removeSearchPin = () => {
    const map = mapInstanceRef.current;
    if (searchMarkerRef.current && map) {
      if (typeof map.hasLayer === "function" && map.hasLayer(searchMarkerRef.current)) {
        map.removeLayer(searchMarkerRef.current);
      }
      searchMarkerRef.current = null;
    }
    if (!bbox) {
      setLiveLocationName(null);
    }
  };

  const clearSearch = () => {
    setSearchQuery("");
    setSearchResults([]);
    setSearchError(null);
    setShowSearchDropdown(false);
    removeSearchPin();
  };

  // Map Rotation Functions (Arbitrary Degrees, Random Degrees, and Steps)
  const setBearingTo = (targetBearing: number) => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const normalized = ((Math.round(targetBearing) % 360) + 360) % 360;
    setBearing(normalized);
    if (typeof map.setBearing === "function") {
      map.setBearing(normalized);
    } else {
      const pane = mapRef.current?.querySelector(".leaflet-map-pane") as HTMLElement;
      if (pane) {
        pane.style.transition = "transform 0.25s cubic-bezier(0.2, 0.8, 0.2, 1)";
        pane.style.transform = `rotate(${normalized}deg)`;
      }
    }
  };

  const rotateMapBy = (delta: number) => {
    setBearingTo(bearing + delta);
  };

  const rotateRandom = () => {
    let randomBearing = Math.floor(Math.random() * 360);
    // Ensure noticeability: change by at least 25 degrees
    if (Math.abs(randomBearing - bearing) < 25) {
      randomBearing = (randomBearing + 90) % 360;
    }
    setBearingTo(randomBearing);
  };

  const resetNorth = () => {
    setBearingTo(0);
  };

  // Compass Drag / Touch Handling to rotate to ANY arbitrary angle
  const calculateBearingFromEvent = (clientX: number, clientY: number) => {
    if (!compassDiscRef.current) return;
    const rect = compassDiscRef.current.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = clientX - cx;
    const dy = clientY - cy;
    // North is straight up (dy < 0, dx = 0 => 0 deg)
    const rad = Math.atan2(dy, dx);
    let deg = Math.round((rad * (180 / Math.PI)) + 90);
    deg = ((deg % 360) + 360) % 360;
    setBearingTo(deg);
  };

  const handleCompassPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    setIsDraggingCompass(true);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    calculateBearingFromEvent(e.clientX, e.clientY);
  };

  const handleCompassPointerMove = (e: React.PointerEvent) => {
    if (!isDraggingCompass) return;
    calculateBearingFromEvent(e.clientX, e.clientY);
  };

  const handleCompassPointerUp = (e: React.PointerEvent) => {
    if (isDraggingCompass) {
      setIsDraggingCompass(false);
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {}
    }
  };

  // Auto-Zoom to Searched Place or Preset
  const zoomToPlace = (lat: number, lon: number, title: string, boundingbox?: string[]) => {
    const map = mapInstanceRef.current;
    if (!map) return;

    setLiveLocationName(title);
    setMousePos({ lat, lng: lon });
    setShowSearchDropdown(false);
    setSearchQuery(title);

    // Remove previous target marker if exists
    if (searchMarkerRef.current && map.hasLayer(searchMarkerRef.current)) {
      map.removeLayer(searchMarkerRef.current);
    }

    // Auto-zoom to bounding box or coordinates
    if (boundingbox && boundingbox.length === 4) {
      const south = parseFloat(boundingbox[0]);
      const north = parseFloat(boundingbox[1]);
      const west = parseFloat(boundingbox[2]);
      const east = parseFloat(boundingbox[3]);
      map.flyToBounds([[south, west], [north, east]], {
        duration: 1.6,
        maxZoom: 15,
        padding: [60, 60],
      });
    } else {
      map.flyTo([lat, lon], 13, { duration: 1.6 });
    }

    // Drop glowing target marker
    if (L) {
      const targetIcon = L.divIcon({
        className: "satquery-search-pulse",
        html: `
          <div class="relative flex items-center justify-center -translate-x-1/2 -translate-y-1/2 pointer-events-none">
            <span class="absolute w-10 h-10 rounded-full bg-[#00F0FF]/30 animate-ping"></span>
            <span class="absolute w-6 h-6 rounded-full border border-[#00F0FF] animate-pulse"></span>
            <span class="w-3.5 h-3.5 rounded-full bg-[#00F0FF] border-2 border-white shadow-[0_0_12px_#00F0FF]"></span>
          </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });

      const marker = L.marker([lat, lon], { icon: targetIcon }).addTo(map);
      marker.bindPopup(
        `<div style="font-family: monospace; font-size: 11px; color: #fff; background: #0b0b0f; border: 1px solid #00F0FF; padding: 6px 10px; border-radius: 8px;">
          <div style="color: #00F0FF; font-weight: bold; text-transform: uppercase; font-size: 9px; letter-spacing: 0.1em;">TARGET ACQUIRED</div>
          <div style="font-weight: 600; margin-top: 2px;">${title}</div>
          <div style="color: rgba(255,255,255,0.6); font-size: 9px; margin-top: 2px;">${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E</div>
        </div>`
      ).openPopup();
      searchMarkerRef.current = marker;
    }
  };

  // Search Submission Handler
  const handleSearch = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const queryToUse = (customQuery !== undefined ? customQuery : searchQuery).trim();
    if (!queryToUse) return;

    // Check if coordinates entered directly (lat, lon)
    const coordMatch = queryToUse.match(/^(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)$/);
    if (coordMatch) {
      const lat = parseFloat(coordMatch[1]);
      const lon = parseFloat(coordMatch[2]);
      if (!isNaN(lat) && !isNaN(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
        zoomToPlace(lat, lon, `Coord [${lat.toFixed(4)}, ${lon.toFixed(4)}]`);
        return;
      }
    }

    setIsSearching(true);
    setSearchError(null);
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(queryToUse)}&limit=6&addressdetails=1`,
        { headers: { "Accept-Language": "en" } }
      );
      if (!res.ok) throw new Error("Search service failed");
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        setSearchResults(data);
        setShowSearchDropdown(true);
        // If user submitted via Enter key or Locate button, auto-zoom to the top result immediately!
        if (e) {
          const first = data[0];
          zoomToPlace(
            parseFloat(first.lat),
            parseFloat(first.lon),
            first.display_name.split(",").slice(0, 3).join(","),
            first.boundingbox
          );
        }
      } else {
        setSearchError("No geographic regions found for query.");
        setShowSearchDropdown(true);
      }
    } catch (err) {
      setSearchError("Geocoding service unavailable.");
      setShowSearchDropdown(true);
    } finally {
      setIsSearching(false);
    }
  };

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

      {/* ── AUTO-ZOOM SEARCH BAR & EO TARGET HUBS ── */}
      <div 
        ref={searchContainerRef}
        className="fixed sm:absolute top-16 sm:top-5 left-3 right-3 sm:left-1/2 sm:-translate-x-1/2 sm:right-auto sm:w-[460px] md:w-[520px] z-[550] pointer-events-auto font-mono"
      >
        <form 
          onSubmit={(e) => handleSearch(e)}
          className="relative flex items-center rounded-2xl sm:rounded-full bg-black/90 backdrop-blur-2xl border border-white/25 shadow-2xl p-1.5 px-3 focus-within:border-[#00F0FF] focus-within:shadow-[0_0_20px_rgba(0,240,255,0.25)] transition-all"
        >
          <Search className="w-4 h-4 text-white/40 ml-1.5 shrink-0" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              const val = e.target.value;
              setSearchQuery(val);
              if (!val.trim()) {
                removeSearchPin();
                setSearchResults([]);
                setSearchError(null);
              } else {
                if (!showSearchDropdown) setShowSearchDropdown(true);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                clearSearch();
              }
            }}
            onFocus={() => setShowSearchDropdown(true)}
            placeholder="Search city, region, or coords (lat, lon)..."
            className="flex-1 bg-transparent px-2.5 py-1 text-xs text-white placeholder-white/40 outline-none font-mono"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={clearSearch}
              className="p-1 text-white/40 hover:text-white transition-colors cursor-pointer mr-1"
              title="Clear search and remove target pin"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            type="submit"
            disabled={isSearching}
            className="px-3.5 py-1.5 rounded-full bg-white hover:bg-[#00F0FF] text-black font-bold text-[10px] uppercase tracking-wider transition-all disabled:opacity-50 flex items-center gap-1 shrink-0 cursor-pointer shadow-md"
          >
            {isSearching ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Scanning</span>
              </>
            ) : (
              <span>Locate</span>
            )}
          </button>
        </form>

        {/* Search Results & Quick Preset Dropdown */}
        {showSearchDropdown && (
          <div className="absolute top-full left-0 w-full mt-2 rounded-2xl bg-black/95 backdrop-blur-2xl border border-white/20 shadow-2xl p-3 max-h-[60vh] overflow-y-auto custom-scrollbar animate-slide-up">
            {/* Geocoded Search Results */}
            {searchResults.length > 0 && (
              <div className="flex flex-col gap-1 mb-3">
                <div className="text-[9px] text-[#00F0FF] font-bold tracking-widest uppercase mb-1 flex items-center justify-between">
                  <span>RESOLVED GEOGRAPHIC TARGETS</span>
                  <span>{searchResults.length} RESULTS</span>
                </div>
                {searchResults.map((result: any, idx: number) => {
                  const title = result.display_name.split(",").slice(0, 3).join(",");
                  const subtitle = result.display_name.split(",").slice(3).join(",");
                  const lat = parseFloat(result.lat);
                  const lon = parseFloat(result.lon);
                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => zoomToPlace(lat, lon, title, result.boundingbox)}
                      className="w-full text-left p-2.5 rounded-xl hover:bg-white/10 border border-transparent hover:border-white/20 transition-all flex items-start gap-2.5 cursor-pointer group"
                    >
                      <MapPin className="w-4 h-4 text-[#00F0FF] shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-semibold text-white group-hover:text-[#00F0FF] truncate">
                          {title}
                        </div>
                        <div className="text-[10px] text-white/50 truncate mt-0.5">
                          {subtitle || result.type}
                        </div>
                        <div className="text-[9px] text-white/40 font-mono mt-0.5">
                          {lat.toFixed(4)}° N, {lon.toFixed(4)}° E
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {searchError && (
              <div className="p-2 mb-2 rounded-lg bg-red-950/40 border border-red-500/40 text-red-300 text-[10px]">
                {searchError}
              </div>
            )}

            {/* Curated EO Target Presets */}
            <div className="flex flex-col gap-1 pt-1 border-t border-white/10">
              <div className="text-[9px] text-white/50 font-bold tracking-widest uppercase mb-1 flex items-center justify-between">
                <span>QUICK TARGET HUBS</span>
                <span className="text-[8px] text-white/40">1-TAP AUTO-ZOOM</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                {TARGET_PRESETS.map((preset) => (
                  <button
                    key={preset.name}
                    type="button"
                    onClick={() => zoomToPlace(preset.lat, preset.lon, preset.name)}
                    className="p-2 rounded-xl bg-white/5 hover:bg-white/15 border border-white/10 hover:border-[#00F0FF]/50 transition-all text-left flex items-start gap-2 cursor-pointer group"
                  >
                    <Crosshair className="w-3.5 h-3.5 text-[#00F0FF] shrink-0 mt-0.5 group-hover:rotate-45 transition-transform" />
                    <div className="min-w-0 flex-1">
                      <div className="text-[11px] font-semibold text-white truncate group-hover:text-[#00F0FF]">
                        {preset.name}
                      </div>
                      <div className="text-[9px] text-white/40 truncate">
                        {preset.desc}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── MAP ROTATION & COMPASS HUD (ARBITRARY & RANDOM DEGREES) ── */}
      <div className="fixed sm:absolute top-32 sm:top-20 right-3 sm:right-5 z-[550] flex flex-col items-end gap-2 pointer-events-auto font-mono">
        {/* Main Compass & Action Toolbar Column */}
        <div className="flex flex-col items-center gap-1.5">
          {/* Draggable Aerospace Compass Dial */}
          <div className="relative select-none" title="Drag compass ring or needle to rotate arbitrary degrees, or double-click to reset North">
            <div
              ref={compassDiscRef}
              onPointerDown={handleCompassPointerDown}
              onPointerMove={handleCompassPointerMove}
              onPointerUp={handleCompassPointerUp}
              onPointerCancel={handleCompassPointerUp}
              onDoubleClick={resetNorth}
              className={`w-14 h-14 rounded-full backdrop-blur-2xl border flex items-center justify-center transition-all shadow-2xl cursor-grab active:cursor-grabbing touch-none select-none relative ${
                isDraggingCompass
                  ? "bg-black/95 border-[#00F0FF] shadow-[0_0_25px_rgba(0,240,255,0.7)] scale-105"
                  : bearing !== 0
                    ? "bg-black/90 border-[#00F0FF]/80 shadow-[0_0_16px_rgba(0,240,255,0.4)]"
                    : "bg-black/80 border-white/20 hover:border-white/50"
              }`}
            >
              {/* Outer Cardinal Graduation Ticks */}
              <div className="absolute inset-0 pointer-events-none">
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[8px] font-bold text-red-400">N</span>
                <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[7px] text-white/50">E</span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[7px] text-white/50">S</span>
                <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[7px] text-white/50">W</span>
              </div>

              {/* Dynamic Rotating Gyro Needle */}
              <div
                className="w-full h-full flex items-center justify-center transition-transform duration-100 pointer-events-none"
                style={{ transform: `rotate(${-bearing}deg)` }}
              >
                <Navigation 
                  className={`w-7 h-7 drop-shadow-md transition-colors ${
                    bearing !== 0 ? "text-[#00F0FF] fill-[#00F0FF]/40" : "text-red-500 fill-red-500/40"
                  }`} 
                />
              </div>

              {/* Center Pivot Point */}
              <div className="absolute w-2 h-2 rounded-full bg-white border border-black shadow pointer-events-none" />

              {/* Exact Degree Readout Badge */}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowRotationFineTune(!showRotationFineTune);
                }}
                className={`absolute -bottom-2.5 px-2 py-0.5 rounded-full border text-[8px] font-bold tracking-tight shadow-md cursor-pointer transition-all ${
                  bearing !== 0
                    ? "bg-black border-[#00F0FF] text-[#00F0FF]"
                    : "bg-black/90 border-white/30 text-white hover:border-white"
                }`}
                title="Click to open Degree Slider & Direct Angle Input"
              >
                {bearing}°
              </button>
            </div>
          </div>

          {/* Quick Action Toolbar (Steps, Random Degree, Slider Toggle, 0° Reset) */}
          <div className="flex flex-col gap-1 bg-black/85 backdrop-blur-xl border border-white/20 p-1 rounded-xl shadow-xl mt-1.5 items-center">
            {/* Step Counter-Clockwise */}
            <button
              type="button"
              onClick={() => rotateMapBy(-45)}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-white/70 hover:text-[#00F0FF] hover:bg-white/10 transition-colors cursor-pointer"
              title="Rotate -45° (↺)"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>

            {/* Step Clockwise */}
            <button
              type="button"
              onClick={() => rotateMapBy(45)}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-white/70 hover:text-[#00F0FF] hover:bg-white/10 transition-colors cursor-pointer"
              title="Rotate +45° (↻)"
            >
              <RotateCw className="w-3.5 h-3.5" />
            </button>

            {/* Random Degrees Button (Roll to random angle) */}
            <button
              type="button"
              onClick={rotateRandom}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-[#00F0FF]/90 hover:text-white hover:bg-[#00F0FF]/20 transition-all cursor-pointer group"
              title="Rotate Random Degrees (🎲 Spin to any angle)"
            >
              <Dices className="w-4 h-4 group-hover:rotate-180 transition-transform duration-300" />
            </button>

            {/* Fine-Tune Slider Toggle */}
            <button
              type="button"
              onClick={() => setShowRotationFineTune(!showRotationFineTune)}
              className={`w-8 h-8 rounded-lg flex items-center justify-center transition-colors cursor-pointer ${
                showRotationFineTune 
                  ? "bg-[#00F0FF] text-black shadow-[0_0_10px_rgba(0,240,255,0.5)]" 
                  : "text-white/70 hover:text-white hover:bg-white/10"
              }`}
              title="Toggle Degree Slider & Arbitrary Angle Control"
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
            </button>

            {/* Quick 0° North Reset Button (visible when tilted) */}
            {bearing !== 0 && (
              <button
                type="button"
                onClick={resetNorth}
                className="w-8 h-6 rounded-md bg-[#00F0FF]/20 border border-[#00F0FF]/50 text-[#00F0FF] text-[8px] font-bold uppercase tracking-wider flex items-center justify-center hover:bg-[#00F0FF] hover:text-black transition-all cursor-pointer shadow-[0_0_8px_rgba(0,240,255,0.3)]"
                title="Reset to 0° North"
              >
                0° N
              </button>
            )}
          </div>
        </div>

        {/* ── EXPANDED ROTATION FINE-TUNE POPOVER (SLIDER + NUMERIC INPUT + PRESETS) ── */}
        {showRotationFineTune && (
          <div 
            ref={rotationFineTuneRef}
            className="w-64 bg-black/95 backdrop-blur-2xl border border-white/20 rounded-2xl p-3.5 shadow-2xl flex flex-col gap-3 animate-slide-up mr-1"
          >
            <div className="flex items-center justify-between border-b border-white/15 pb-2">
              <div className="flex items-center gap-1.5 text-xs font-bold tracking-wider text-[#00F0FF] uppercase">
                <Compass className="w-3.5 h-3.5" />
                <span>Arbitrary Bearing</span>
              </div>
              <button
                type="button"
                onClick={() => setShowRotationFineTune(false)}
                className="text-white/40 hover:text-white transition-colors cursor-pointer p-0.5"
                title="Close rotation controls"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Direct Degree Number Input & Display */}
            <div className="flex items-center justify-between bg-white/5 border border-white/15 rounded-xl px-3 py-2">
              <span className="text-[10px] text-white/50 tracking-wider uppercase">Custom Degree</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min="0"
                  max="359"
                  value={bearing}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setBearingTo(isNaN(val) ? 0 : val);
                  }}
                  className="w-14 bg-black/70 border border-white/20 rounded px-1.5 py-0.5 text-right font-bold text-xs text-[#00F0FF] outline-none focus:border-[#00F0FF]"
                />
                <span className="text-xs font-bold text-white/60">°</span>
              </div>
            </div>

            {/* Continuous 0° - 359° Degree Slider */}
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-[9px] text-white/40 font-mono">
                <span>0° (N)</span>
                <span>90°</span>
                <span>180° (S)</span>
                <span>270°</span>
                <span>359°</span>
              </div>
              <input
                type="range"
                min="0"
                max="359"
                step="1"
                value={bearing}
                onChange={(e) => setBearingTo(parseInt(e.target.value))}
                className="w-full h-2 accent-[#00F0FF] bg-white/10 rounded-lg cursor-pointer"
              />
            </div>

            {/* Quick Cardinal Presets */}
            <div className="grid grid-cols-4 gap-1 pt-1 border-t border-white/10">
              <button
                type="button"
                onClick={() => setBearingTo(0)}
                className={`py-1 rounded text-[9px] font-bold tracking-wider transition-all cursor-pointer ${
                  bearing === 0 ? "bg-[#00F0FF] text-black" : "bg-white/5 hover:bg-white/15 text-white/80"
                }`}
              >
                0° N
              </button>
              <button
                type="button"
                onClick={() => setBearingTo(90)}
                className={`py-1 rounded text-[9px] font-bold tracking-wider transition-all cursor-pointer ${
                  bearing === 90 ? "bg-[#00F0FF] text-black" : "bg-white/5 hover:bg-white/15 text-white/80"
                }`}
              >
                90° E
              </button>
              <button
                type="button"
                onClick={() => setBearingTo(180)}
                className={`py-1 rounded text-[9px] font-bold tracking-wider transition-all cursor-pointer ${
                  bearing === 180 ? "bg-[#00F0FF] text-black" : "bg-white/5 hover:bg-white/15 text-white/80"
                }`}
              >
                180° S
              </button>
              <button
                type="button"
                onClick={() => setBearingTo(270)}
                className={`py-1 rounded text-[9px] font-bold tracking-wider transition-all cursor-pointer ${
                  bearing === 270 ? "bg-[#00F0FF] text-black" : "bg-white/5 hover:bg-white/15 text-white/80"
                }`}
              >
                270° W
              </button>
            </div>

            {/* Roll Random Degree Button in popover */}
            <button
              type="button"
              onClick={rotateRandom}
              className="w-full py-2 rounded-xl bg-gradient-to-r from-[#00F0FF]/20 to-cyan-500/20 hover:from-[#00F0FF]/30 hover:to-cyan-500/30 border border-[#00F0FF]/40 text-[#00F0FF] text-[10px] font-bold tracking-wider uppercase flex items-center justify-center gap-2 cursor-pointer transition-all shadow-sm"
            >
              <Dices className="w-3.5 h-3.5" />
              <span>Rotate Random Degrees</span>
            </button>
          </div>
        )}
      </div>

      {/* LAT/LNG TRACKER & LOCATION */}
      <div className="absolute bottom-20 sm:bottom-6 right-3 sm:right-16 z-[400] pointer-events-none flex flex-col items-end gap-2">
        {liveLocationName && (
          <div className="bg-black/80 backdrop-blur-md border border-[#00F0FF]/30 px-3 py-2 text-[10px] tracking-widest text-[#00F0FF] rounded-lg max-w-[250px] text-right truncate">
            {liveLocationName}
          </div>
        )}
        <div className="bg-black/70 backdrop-blur-md border border-white/10 px-2.5 sm:px-4 py-1.5 sm:py-2 text-[9px] sm:text-[10px] tracking-[0.15em] sm:tracking-[0.2em] text-[#00F0FF] rounded-lg">
          LAT: {mousePos.lat.toFixed(3)} / LNG: {mousePos.lng.toFixed(3)} {bearing !== 0 && `// BRG: ${bearing}°`}
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
