"use client";

import React, { useEffect, useRef, useState } from 'react';
import Globe from 'react-globe.gl';

export default function RealGlobe() {
  const globeRef = useRef<any>();
  const [dimensions, setDimensions] = useState({ width: 800, height: 800 });
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    
    // Set initial dimensions based on window size
    const handleResize = () => {
      const containerWidth = window.innerWidth;
      const size = Math.min(containerWidth, 800);
      setDimensions({ width: size, height: size });
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (isMounted && globeRef.current) {
      globeRef.current.controls().autoRotate = true;
      globeRef.current.controls().autoRotateSpeed = 0.5;
      globeRef.current.controls().enableZoom = true;
    }
  }, [isMounted]);

  const markers = [
    { lat: 37.7595, lng: -122.4367, size: 0.05, color: '#ff3366' },
    { lat: 40.7128, lng: -74.0060, size: 0.05, color: '#ff3366' },
    { lat: 51.5074, lng: -0.1278, size: 0.05, color: '#ff3366' },
    { lat: 35.6895, lng: 139.6917, size: 0.05, color: '#ff3366' }
  ];

  if (!isMounted) return <div style={{ width: 800, height: 800, maxWidth: "100%", aspectRatio: 1 }} />;

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", cursor: "grab", overflow: "hidden" }}>
      <Globe
        ref={globeRef}
        width={dimensions.width}
        height={dimensions.height}
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        backgroundColor="rgba(0,0,0,0)"
        atmosphereColor="lightskyblue"
        atmosphereAltitude={0.25}
        pointsData={markers}
        pointAltitude={0.01}
        pointColor="color"
        pointRadius="size"
        pointsMerge={false}
      />
    </div>
  );
}
