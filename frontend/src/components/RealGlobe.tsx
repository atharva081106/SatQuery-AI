"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";

export default function RealGlobe() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    // Get strict 1:1 square dimension
    const getSquareSize = () => {
      const rect = canvas.getBoundingClientRect();
      const size = Math.round(rect.width || rect.height || 800);
      return Math.max(300, size);
    };

    let size = getSquareSize();

    // Scene & Camera - strict 1:1 aspect ratio ensures a 100% perfect circle
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    camera.position.set(0, 0, 5.0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(size, size);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 2.4);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.4);
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
    const earthRadius = 1.82;
    const earthGeometry = new THREE.SphereGeometry(earthRadius, 64, 64);
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: earthDay,
      roughness: 0.65,
      metalness: 0.08
    });
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    globeGroup.add(earthMesh);

    // 2. Cloud Sphere (slightly larger)
    const cloudsGeometry = new THREE.SphereGeometry(earthRadius + 0.02, 64, 64);
    const cloudsMaterial = new THREE.MeshStandardMaterial({
      map: earthClouds,
      transparent: true,
      opacity: 0.38,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const cloudsMesh = new THREE.Mesh(cloudsGeometry, cloudsMaterial);
    globeGroup.add(cloudsMesh);

    // 3. Atmosphere Glow Outer Mesh
    const glowGeometry = new THREE.SphereGeometry(earthRadius + 0.16, 64, 64);
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
          float intensity = pow(0.68 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          gl_FragColor = vec4(0.22, 0.68, 1.0, 1.0) * intensity * 0.95;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false
    });
    const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
    scene.add(glowMesh);

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

    // Strict 1:1 Responsive Resize Handler
    const handleResize = () => {
      if (!canvas) return;
      const newSize = getSquareSize();
      if (newSize !== size) {
        size = newSize;
        camera.aspect = 1; // Strict 1:1 circle aspect
        camera.updateProjectionMatrix();
        renderer.setSize(size, size);
      }
    };

    const resizeObserver = new ResizeObserver(() => handleResize());
    resizeObserver.observe(container);

    // Animation Loop
    let animationFrameId: number;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      // Continuous auto-rotation when not dragging
      if (!isDragging) {
        targetRotationY += 0.0016;
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
      cloudsMesh.rotation.y += 0.0005;

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId);
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
      {/* Strict 1:1 aspect-ratio container guarantees a perfectly round sphere */}
      <div className="relative flex items-center justify-center w-[850px] h-[850px] max-w-[90vw] max-h-[85vh] aspect-square">
        <canvas ref={canvasRef} className="block w-full h-full aspect-square" />
      </div>
    </div>
  );
}
