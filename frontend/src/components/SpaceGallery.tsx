"use client";

import React, { useRef, useState, useMemo, Suspense } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Stars, Html, useTexture } from '@react-three/drei';
import * as THREE from 'three';

// Constants for orbit (stylized scale to fit nicely on screen)
const SCALING = 0.5;
const PLANETS = [
  { name: 'Mercury', texture: '/textures/2k_mercury.jpg', size: 0.38 * SCALING, distance: 3, speed: 0.04, color: '#b0b0b0' },
  { name: 'Venus', texture: '/textures/2k_venus_surface.jpg', size: 0.95 * SCALING, distance: 5, speed: 0.015, color: '#e3bb76' },
  { name: 'Earth', texture: '/textures/2k_earth_daymap.jpg', size: 1 * SCALING, distance: 7, speed: 0.01, color: '#2b82c9' },
  { name: 'Mars', texture: '/textures/2k_mars.jpg', size: 0.53 * SCALING, distance: 9, speed: 0.008, color: '#c1440e' },
  { name: 'Jupiter', texture: '/textures/2k_jupiter.jpg', size: 2.5 * SCALING, distance: 13, speed: 0.002, color: '#d39c7e' },
  { name: 'Saturn', texture: '/textures/2k_saturn.jpg', size: 2 * SCALING, distance: 18, speed: 0.0009, ring: '/textures/2k_saturn_ring_alpha.png', color: '#c5ab6e' },
  { name: 'Uranus', texture: '/textures/2k_uranus.jpg', size: 1.2 * SCALING, distance: 23, speed: 0.0004, color: '#4b70dd' },
  { name: 'Neptune', texture: '/textures/2k_neptune.jpg', size: 1.1 * SCALING, distance: 28, speed: 0.0001, color: '#274687' },
];

function OrbitRing({ distance }: { distance: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[distance - 0.02, distance + 0.02, 64]} />
      <meshBasicMaterial color="#ffffff" opacity={0.1} transparent side={THREE.DoubleSide} />
    </mesh>
  );
}

function Planet({ data }: { data: typeof PLANETS[0] }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const orbitRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  // useTexture from drei simplifies loading and caches
  const texture = useTexture(data.texture);
  const ringTexture = data.ring ? useTexture(data.ring) : null;

  useFrame((state, delta) => {
    if (orbitRef.current) {
      orbitRef.current.rotation.y += data.speed * (delta * 60);
    }
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01 * (delta * 60);
    }
  });

  return (
    <group>
      <OrbitRing distance={data.distance} />
      <group ref={orbitRef}>
        <mesh 
          ref={meshRef} 
          position={[data.distance, 0, 0]}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
            document.body.style.cursor = 'pointer';
          }}
          onPointerOut={() => {
            setHovered(false);
            document.body.style.cursor = 'auto';
          }}
        >
          <sphereGeometry args={[data.size, 64, 64]} />
          <meshStandardMaterial 
            map={texture} 
            roughness={0.7} 
            metalness={0.1} 
            emissive={new THREE.Color(0x333333)}
          />
          
          {data.ring && ringTexture && (
            <mesh rotation={[-Math.PI / 2 + 0.2, 0, 0]}>
              <ringGeometry args={[data.size * 1.4, data.size * 2.2, 64]} />
              <meshStandardMaterial 
                map={ringTexture} 
                transparent 
                opacity={0.8} 
                side={THREE.DoubleSide}
              />
            </mesh>
          )}

          <Html distanceFactor={15}>
            <div className={`transition-opacity duration-300 font-mono text-xs px-2 py-1 bg-black/80 text-white border border-white/20 whitespace-nowrap rounded-sm pointer-events-none ${hovered ? 'opacity-100' : 'opacity-0'}`}>
              {data.name}
            </div>
          </Html>
        </mesh>
      </group>
    </group>
  );
}

function Sun() {
  const texture = useTexture('/textures/2k_sun.jpg');
  const sunRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (sunRef.current) {
      sunRef.current.rotation.y += 0.002 * (delta * 60);
    }
  });

  return (
    <mesh ref={sunRef}>
      <sphereGeometry args={[1.5, 64, 64]} />
      <meshBasicMaterial map={texture} color="#ffffee" />
      <pointLight color="#ffffff" intensity={15} distance={200} decay={1.0} />
    </mesh>
  );
}

export default function SpaceGallery() {
  return (
    <div className="w-full h-[100dvh] bg-black relative z-10 overflow-hidden flex flex-col">
      <div className="absolute top-8 left-1/2 -translate-x-1/2 z-20 text-center pointer-events-none">
        <h2 className="text-white text-xl md:text-3xl font-mono tracking-[0.2em] font-bold uppercase drop-shadow-md">
          Solar System Explorer
        </h2>
        <p className="text-white/60 text-xs mt-2 font-mono uppercase tracking-widest">
          Interactive 3D WebGL Simulation
        </p>
      </div>
      
      <div className="absolute inset-0 z-0">
        <Suspense fallback={
          <div className="flex items-center justify-center w-full h-full text-white font-mono text-sm tracking-widest">
            INITIALIZING 3D ENGINE...
          </div>
        }>
          <Canvas camera={{ position: [0, 15, 30], fov: 45 }}>
            <color attach="background" args={['#000005']} />
            <ambientLight intensity={1.5} />
            
            <Sun />
            {PLANETS.map((planet) => (
              <Planet key={planet.name} data={planet} />
            ))}

            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
            <OrbitControls 
              enablePan={true} 
              enableZoom={true} 
              enableRotate={true}
              minDistance={3}
              maxDistance={80}
            />
          </Canvas>
        </Suspense>
      </div>

      <div className="absolute bottom-6 left-0 right-0 z-20 text-center pointer-events-none">
        <p className="text-white/50 text-[10px] sm:text-xs tracking-widest uppercase font-mono bg-black/40 inline-block px-4 py-2 rounded-full backdrop-blur-sm border border-white/10">
          ( Touch &amp; Drag to Orbit • Scroll to Zoom )
        </p>
      </div>
    </div>
  );
}
