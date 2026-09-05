"use client";
import { useEffect, useRef } from "react";

interface GeoJSONMapOverlayProps {
  geojsonData: any;
}

const CLASS_COLORS: Record<string, string> = {
  "Water Bodies":  "#22d3ee",
  "Water":         "#22d3ee",
  "Vegetation":    "#4ade80",
  "Forest":        "#22c55e",
  "Built-up":      "#f97316",
  "Built-Up":      "#f97316",
  "Bare Soil":     "#a16207",
  "Cloud":         "#94a3b8",
  "Background":    "#475569",
  "Land":          "#86efac",
};

function getColor(label: string) {
  for (const [key, color] of Object.entries(CLASS_COLORS)) {
    if (label.toLowerCase().includes(key.toLowerCase())) return color;
  }
  return "#ffffff";
}

export default function GeoJSONMapOverlay({ geojsonData }: GeoJSONMapOverlayProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (!mapRef.current || !geojsonData) return;

    // Clean up previous instance
    if (mapInstanceRef.current) {
      mapInstanceRef.current.remove();
      mapInstanceRef.current = null;
    }
    const el = mapRef.current as any;
    if (el._leaflet_id) {
      delete el._leaflet_id;
    }

    const init = async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet/dist/leaflet.css");

      // Compute bounds from GeoJSON
      let bounds: any = null;

      const map = L.map(mapRef.current!, {
        zoomControl: true,
        attributionControl: false,
        scrollWheelZoom: false,
      });
      mapInstanceRef.current = map;

      // Bhuvan as default base layer for the results map
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 18, attribution: "" }
      ).addTo(map);

      // Render GeoJSON
      try {
        const geoLayer = L.geoJSON(geojsonData, {
          style: (feature) => {
            const cls = feature?.properties?.class_name || "";
            const color = getColor(cls);
            return {
              color: color,
              weight: 2,
              opacity: 0.9,
              fillColor: color,
              fillOpacity: 0.25,
            };
          },
          onEachFeature: (feature, layer) => {
            const p = feature.properties || {};
            const cls = p.class_name || "Feature";
            const area = p.area_km2 != null ? `${Number(p.area_km2).toFixed(3)} km²` : "";
            const pct = p.coverage_pct != null ? `${Number(p.coverage_pct).toFixed(1)}%` : "";
            layer.bindPopup(
              `<div style="font-family:monospace;font-size:11px;color:#111">` +
              `<strong>${cls}</strong>${area ? `<br/>Area: ${area}` : ""}${pct ? `<br/>Coverage: ${pct}` : ""}` +
              `</div>`,
              { maxWidth: 180 }
            );
          },
        }).addTo(map);

        bounds = geoLayer.getBounds();
        if (bounds.isValid()) {
          map.fitBounds(bounds, { padding: [20, 20], maxZoom: 14 });
        } else {
          map.setView([20.5937, 78.9629], 5); // India center
        }
      } catch (e) {
        map.setView([20.5937, 78.9629], 5);
      }
    };

    init();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [geojsonData]);

  if (!geojsonData) return null;

  return (
    <div className="flex flex-col gap-2 mb-6">
      <div className="micro-cap text-white/50 mb-1 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        GIS VECTOR MAP OVERLAY
        <span className="text-emerald-400 text-[10px]">• RFC 7946 WGS84</span>
      </div>
      <div
        ref={mapRef}
        style={{ height: "220px", borderRadius: "6px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.12)" }}
      />
      <div className="text-[10px] text-white/40 font-mono tracking-wider">
        Click polygons for class info • Compatible with ISRO Bhuvan / QGIS / ArcGIS
      </div>
    </div>
  );
}

