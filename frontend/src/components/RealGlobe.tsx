"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

interface Marker {
  lat: number;
  lng: number;
  name: string;
  color: string;
}

const MARKERS: Marker[] = [
  { lat: 13.72, lng: 80.23, name: "SDSC SHAR (Sriharikota)", color: "#00f0ff" },
  { lat: 12.97, lng: 77.59, name: "ISTRAC (Bengaluru)", color: "#00ff88" },
  { lat: 28.39, lng: -80.6, name: "Cape Canaveral", color: "#ffaa00" },
  { lat: 78.22, lng: 15.65, name: "SvalSat (Arctic Ground)", color: "#ff3366" },
  { lat: 35.68, lng: 139.69, name: "Tokyo Tracking", color: "#b388ff" }
];

function latLngToVector3(lat: number, lng: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 180) * (Math.PI / 180);
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}

export default function RealGlobe() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeStation, setActiveStation] = useState<string>("SDSC SHAR (Sriharikota)");

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    let width = container.clientWidth || 800;
    let height = container.clientHeight || 800;

    // Scene & Camera
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 5.2);

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
    rimLight.position.set(-5, -2, -3);
    scene.add(rimLight);

    // Globe Group
    const globeGroup = new THREE.Group();
    // Tilt Earth slightly on its natural 23.5-degree axis
    globeGroup.rotation.z = (23.5 * Math.PI) / 180;
    scene.add(globeGroup);

    // Textures
    const textureLoader = new THREE.TextureLoader();
    const earthDay = textureLoader.load("/textures/2k_earth_daymap.jpg");
    earthDay.colorSpace = THREE.SRGBColorSpace;

    const earthClouds = textureLoader.load("/textures/2k_earth_clouds.jpg");
    earthClouds.colorSpace = THREE.SRGBColorSpace;

    // 1. Earth Sphere
    const earthRadius = 1.85;
    const earthGeometry = new THREE.SphereGeometry(earthRadius, 64, 64);
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: earthDay,
      roughness: 0.7,
      metalness: 0.1
    });
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    globeGroup.add(earthMesh);

    // 2. Cloud Sphere (slightly larger)
    const cloudsGeometry = new THREE.SphereGeometry(earthRadius + 0.025, 64, 64);
    const cloudsMaterial = new THREE.MeshStandardMaterial({
      map: earthClouds,
      transparent: true,
      opacity: 0.42,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const cloudsMesh = new THREE.Mesh(cloudsGeometry, cloudsMaterial);
    globeGroup.add(cloudsMesh);

    // 3. Atmosphere Glow Outer Mesh
    const glowGeometry = new THREE.SphereGeometry(earthRadius + 0.14, 48, 48);
    const glowMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.68 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.2);
          gl_FragColor = vec4(0.12, 0.65, 1.0, 1.0) * intensity * 0.85;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false
    });
    const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
    scene.add(glowMesh);

    // 4. Ground Station Markers
    const markerRings: THREE.Mesh[] = [];
    MARKERS.forEach((m) => {
      const pos = latLngToVector3(m.lat, m.lng, earthRadius + 0.015);
      
      // Pin dot
      const pinGeo = new THREE.SphereGeometry(0.025, 16, 16);
      const pinMat = new THREE.MeshBasicMaterial({ color: m.color });
      const pinMesh = new THREE.Mesh(pinGeo, pinMat);
      pinMesh.position.copy(pos);
      globeGroup.add(pinMesh);

      // Pulsing telemetry ring
      const ringGeo = new THREE.RingGeometry(0.035, 0.055, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: m.color,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.position.copy(pos);
      ringMesh.lookAt(pos.clone().multiplyScalar(2));
      globeGroup.add(ringMesh);
      markerRings.push(ringMesh);
    });

    // Interaction & Animation State
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let targetRotationX = 0;
    let targetRotationY = 0;
    let velocityX = 0;
    let velocityY = 0;

    const onPointerDown = (e: PointerEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
      velocityX = 0;
      velocityY = 0;
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;

      velocityX = deltaX * 0.005;
      velocityY = deltaY * 0.005;

      targetRotationY += velocityX;
      targetRotationX += velocityY;

      // Limit vertical tilt
      targetRotationX = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, targetRotationX));

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onPointerUp = () => {
      isDragging = false;
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    // Responsive Resize Handler
    const handleResize = () => {
      if (!container) return;
      width = container.clientWidth || 800;
      height = container.clientHeight || 800;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const resizeObserver = new ResizeObserver(() => handleResize());
    resizeObserver.observe(container);

    // Animation Loop
    let animationFrameId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Continuous auto-rotation when not dragging
      if (!isDragging) {
        targetRotationY += 0.0018;
        // Damp inertia
        velocityX *= 0.94;
        velocityY *= 0.94;
        targetRotationY += velocityX;
        targetRotationX += velocityY;
      }

      // Smooth interpolation
      globeGroup.rotation.y += (targetRotationY - globeGroup.rotation.y) * 0.08;
      globeGroup.rotation.x += (targetRotationX - globeGroup.rotation.x) * 0.08;

      // Rotate clouds slightly faster than earth
      cloudsMesh.rotation.y += 0.0006;

      // Pulse telemetry rings
      const pulseScale = 1 + 0.35 * Math.sin(elapsedTime * 4.5);
      markerRings.forEach((ring) => {
        ring.scale.set(pulseScale, pulseScale, 1);
      });

      renderer.render(scene, camera);
    };

    animate();

    // Rotate station telemetry every 4 seconds
    let stationIndex = 0;
    const stationTimer = setInterval(() => {
      stationIndex = (stationIndex + 1) % MARKERS.length;
      setActiveStation(MARKERS[stationIndex].name);
    }, 3800);

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId);
      clearInterval(stationTimer);
      resizeObserver.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);

      earthGeometry.dispose();
      earthMaterial.dispose();
      cloudsGeometry.dispose();
      cloudsMaterial.dispose();
      glowGeometry.dispose();
      glowMaterial.dispose();
      earthDay.dispose();
      earthClouds.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex items-center justify-center cursor-grab active:cursor-grabbing select-none overflow-hidden"
    >
      <canvas ref={canvasRef} className="block w-full h-full max-w-[1000px] max-h-[1000px]" />
      
      {/* Dynamic Telemetry HUD Pin Badge */}
      <div className="absolute top-6 left-6 pointer-events-none hidden sm:flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-black/60 border border-white/15 backdrop-blur-md">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
        <span className="text-[11px] font-mono tracking-wider text-white/80 uppercase">
          TELEMETRY LOCK: <strong className="text-white font-bold">{activeStation}</strong>
        </span>
      </div>

      {/* Orbit Interaction Hint */}
      <div className="absolute bottom-6 right-6 pointer-events-none hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm text-[10px] font-mono text-white/50 tracking-wider">
        <span>⟳ DRAG TO ORBIT</span>
      </div>
    </div>
  );
}
