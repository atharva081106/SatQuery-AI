"use client";

import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ScrollControls, useScroll, Html, Stars } from '@react-three/drei';
import * as THREE from 'three';

function SatelliteParts() {
  const scroll = useScroll();
  
  // References for animation
  const groupRef = useRef<THREE.Group>(null);
  const leftPanelRef = useRef<THREE.Mesh>(null);
  const rightPanelRef = useRef<THREE.Mesh>(null);
  const dishRef = useRef<THREE.Mesh>(null);
  const antennaRef = useRef<THREE.Mesh>(null);
  
  // HTML Opacity Ref
  const htmlRef1 = useRef<HTMLDivElement>(null);
  const htmlRef2 = useRef<HTMLDivElement>(null);
  const htmlRef3 = useRef<HTMLDivElement>(null);
  const htmlRef4 = useRef<HTMLDivElement>(null);

  useFrame((state, delta) => {
    // scroll.offset goes from 0 to 1
    const offset = scroll.offset;
    
    // Smoothly rotate the whole satellite
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.005 * (delta * 60);
      groupRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }

    // Explode components based on scroll
    // The easing factor makes it look smooth
    const explodeAmount = offset * 4; 

    if (leftPanelRef.current) {
      leftPanelRef.current.position.x = THREE.MathUtils.lerp(
        leftPanelRef.current.position.x,
        -1.5 - explodeAmount,
        0.1
      );
    }
    if (rightPanelRef.current) {
      rightPanelRef.current.position.x = THREE.MathUtils.lerp(
        rightPanelRef.current.position.x,
        1.5 + explodeAmount,
        0.1
      );
    }
    if (dishRef.current) {
      dishRef.current.position.y = THREE.MathUtils.lerp(
        dishRef.current.position.y,
        1.2 + explodeAmount * 0.8,
        0.1
      );
    }
    if (antennaRef.current) {
      antennaRef.current.position.y = THREE.MathUtils.lerp(
        antennaRef.current.position.y,
        -1.5 - explodeAmount * 0.8,
        0.1
      );
    }

    // Fade in HTML tooltips when scroll is > 0.8
    const opacity = offset > 0.8 ? (offset - 0.8) * 5 : 0;
    if (htmlRef1.current) htmlRef1.current.style.opacity = opacity.toString();
    if (htmlRef2.current) htmlRef2.current.style.opacity = opacity.toString();
    if (htmlRef3.current) htmlRef3.current.style.opacity = opacity.toString();
    if (htmlRef4.current) htmlRef4.current.style.opacity = opacity.toString();
  });

  return (
    <group ref={groupRef}>
      {/* MAIN BODY (Bus) */}
      <mesh>
        <cylinderGeometry args={[0.8, 0.8, 2, 16]} />
        <meshStandardMaterial color="#c0c0c0" metalness={0.8} roughness={0.2} />
        <Html distanceFactor={10}>
          <div ref={htmlRef1} className="text-white font-mono text-xs bg-black/80 px-2 py-1 border border-white/20 whitespace-nowrap opacity-0 transition-opacity">
            MAIN BUS (Central Processing)
          </div>
        </Html>
      </mesh>

      {/* LEFT SOLAR PANEL */}
      <mesh ref={leftPanelRef} position={[-1.5, 0, 0]}>
        <boxGeometry args={[2, 4, 0.1]} />
        <meshStandardMaterial color="#1a3b5c" metalness={0.5} roughness={0.1} />
        {/* Solar grid pattern (stylized with a grid helper) */}
        <gridHelper args={[2, 10, 0x000000, 0x000000]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.06]} />
        
        <Html distanceFactor={10}>
          <div ref={htmlRef2} className="text-white font-mono text-xs bg-black/80 px-2 py-1 border border-white/20 whitespace-nowrap opacity-0 transition-opacity">
            PORT SOLAR ARRAY (Power)
          </div>
        </Html>
      </mesh>

      {/* RIGHT SOLAR PANEL */}
      <mesh ref={rightPanelRef} position={[1.5, 0, 0]}>
        <boxGeometry args={[2, 4, 0.1]} />
        <meshStandardMaterial color="#1a3b5c" metalness={0.5} roughness={0.1} />
        <gridHelper args={[2, 10, 0x000000, 0x000000]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.06]} />
        
        <Html distanceFactor={10}>
          <div ref={htmlRef3} className="text-white font-mono text-xs bg-black/80 px-2 py-1 border border-white/20 whitespace-nowrap opacity-0 transition-opacity">
            STARBOARD SOLAR ARRAY (Power)
          </div>
        </Html>
      </mesh>

      {/* COMMS DISH */}
      <mesh ref={dishRef} position={[0, 1.2, 0]}>
        {/* A dish shape */}
        <cylinderGeometry args={[1.2, 0.2, 0.5, 16]} />
        <meshStandardMaterial color="#ffffff" metalness={0.3} roughness={0.6} />
        
        <Html distanceFactor={10}>
          <div ref={htmlRef4} className="text-white font-mono text-xs bg-black/80 px-2 py-1 border border-white/20 whitespace-nowrap opacity-0 transition-opacity">
            HIGH-GAIN ANTENNA (Comms)
          </div>
        </Html>
      </mesh>

      {/* BOTTOM ANTENNA */}
      <mesh ref={antennaRef} position={[0, -1.5, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 2, 8]} />
        <meshStandardMaterial color="#ffaa00" metalness={0.9} roughness={0.1} />
      </mesh>
    </group>
  );
}

export default function SatelliteExplodedView() {
  return (
    <div className="w-full h-[100dvh] bg-[#000005] relative flex flex-col">
      <div className="absolute top-8 w-full z-20 text-center pointer-events-none px-4">
        <h2 className="text-white text-xl md:text-3xl font-mono tracking-[0.2em] font-bold uppercase drop-shadow-md">
          Satellite Engineering
        </h2>
        <p className="text-white/60 text-xs mt-2 font-mono uppercase tracking-widest">
          Scroll down to deconstruct
        </p>
      </div>

      <Canvas camera={{ position: [0, 0, 15], fov: 45 }}>
        <ambientLight intensity={1} />
        <directionalLight position={[10, 10, 10]} intensity={2} />
        <pointLight position={[-10, -10, -10]} intensity={1} color="#4b70dd" />
        
        <Stars radius={50} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        
        <ScrollControls pages={3} damping={0.2}>
          <SatelliteParts />
        </ScrollControls>
      </Canvas>
    </div>
  );
}
