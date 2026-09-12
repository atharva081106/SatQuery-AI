"use client";

import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ScrollControls, useScroll, Html, Stars, useTexture } from '@react-three/drei';
import * as THREE from 'three';

function LaunchAnimation() {
  const scroll = useScroll();
  
  // References
  const earthRef = useRef<THREE.Mesh>(null);
  const rocketGroupRef = useRef<THREE.Group>(null);
  const exhaustRef = useRef<THREE.Mesh>(null);
  
  const fairingLeftRef = useRef<THREE.Mesh>(null);
  const fairingRightRef = useRef<THREE.Mesh>(null);
  
  const satelliteGroupRef = useRef<THREE.Group>(null);
  const leftPanelRef = useRef<THREE.Mesh>(null);
  const rightPanelRef = useRef<THREE.Mesh>(null);
  
  const htmlRef = useRef<HTMLDivElement>(null);

  const earthTexture = useTexture('/textures/2k_earth_daymap.jpg');

  useFrame((state, delta) => {
    const offset = scroll.offset; // 0 to 1
    
    // Rotate Earth slowly
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.05 * delta;
    }

    // ANIMATION STAGES

    // Stage 1: Liftoff & Earth Descent (offset 0.1 to 0.4)
    const liftoffProgress = Math.max(0, Math.min(1, (offset - 0.1) / 0.3));
    
    if (earthRef.current) {
      // Earth shrinks and drops
      const targetScale = 1 - (liftoffProgress * 0.9);
      earthRef.current.scale.set(targetScale, targetScale, targetScale);
      earthRef.current.position.y = -20 - (liftoffProgress * 40);
    }
    
    if (exhaustRef.current) {
      // Exhaust appears during liftoff
      if (liftoffProgress > 0 && liftoffProgress < 1) {
        exhaustRef.current.scale.set(1, 1 + Math.random() * 0.5, 1);
        exhaustRef.current.visible = true;
      } else {
        exhaustRef.current.visible = false;
      }
    }
    
    if (rocketGroupRef.current) {
      // Rocket shakes during liftoff
      if (liftoffProgress > 0 && liftoffProgress < 1) {
        rocketGroupRef.current.position.x = (Math.random() - 0.5) * 0.05;
      } else {
        rocketGroupRef.current.position.x = 0;
      }
    }

    // Stage 2: Fairing Separation (offset 0.45 to 0.65)
    const fairingProgress = Math.max(0, Math.min(1, (offset - 0.45) / 0.2));
    
    if (fairingLeftRef.current) {
      fairingLeftRef.current.position.x = -fairingProgress * 5;
      fairingLeftRef.current.rotation.z = fairingProgress * 0.5;
    }
    if (fairingRightRef.current) {
      fairingRightRef.current.position.x = fairingProgress * 5;
      fairingRightRef.current.rotation.z = -fairingProgress * 0.5;
    }

    // Stage 3: Satellite Deployment (offset 0.65 to 0.9)
    const deployProgress = Math.max(0, Math.min(1, (offset - 0.65) / 0.25));
    
    if (satelliteGroupRef.current) {
      // Satellite rises out of the rocket body
      satelliteGroupRef.current.position.y = 1 + (deployProgress * 3);
      // Satellite rotates slowly once deployed
      if (deployProgress > 0) {
        satelliteGroupRef.current.rotation.y += 0.5 * delta;
      }
    }
    
    // Solar panels unfold horizontally
    if (leftPanelRef.current) {
      leftPanelRef.current.position.x = THREE.MathUtils.lerp(0, -1.2, deployProgress);
    }
    if (rightPanelRef.current) {
      rightPanelRef.current.position.x = THREE.MathUtils.lerp(0, 1.2, deployProgress);
    }

    // Stage 4: Mission Success HTML Fade In
    if (htmlRef.current) {
      const opacity = offset > 0.9 ? (offset - 0.9) * 10 : 0;
      htmlRef.current.style.opacity = Math.min(1, opacity).toString();
    }
  });

  return (
    <group>
      {/* EARTH */}
      <mesh ref={earthRef} position={[0, -20, 0]}>
        <sphereGeometry args={[18, 64, 64]} />
        <meshStandardMaterial map={earthTexture} roughness={0.6} />
      </mesh>

      {/* ROCKET OVERALL GROUP */}
      <group ref={rocketGroupRef} position={[0, -1, 0]}>
        
        {/* ROCKET BODY (Stage 1 / 2) */}
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[0.5, 0.5, 4, 32]} />
          <meshStandardMaterial color="#eeeeee" metalness={0.2} roughness={0.5} />
        </mesh>
        
        {/* EXHAUST FLAME */}
        <mesh ref={exhaustRef} position={[0, -2.5, 0]} visible={false}>
          <coneGeometry args={[0.4, 1.5, 16]} />
          <meshBasicMaterial color="#ff7700" />
        </mesh>

        {/* FAIRING LEFT */}
        <mesh ref={fairingLeftRef} position={[0, 2.5, 0]}>
          <cylinderGeometry args={[0.01, 0.5, 1, 16, 1, false, Math.PI, Math.PI]} />
          <meshStandardMaterial color="#dddddd" side={THREE.DoubleSide} />
        </mesh>

        {/* FAIRING RIGHT */}
        <mesh ref={fairingRightRef} position={[0, 2.5, 0]}>
          <cylinderGeometry args={[0.01, 0.5, 1, 16, 1, false, 0, Math.PI]} />
          <meshStandardMaterial color="#dddddd" side={THREE.DoubleSide} />
        </mesh>

        {/* PAYLOAD (Satellite) */}
        <group ref={satelliteGroupRef} position={[0, 1, 0]} scale={[0.4, 0.4, 0.4]}>
          {/* Main Bus */}
          <mesh>
            <cylinderGeometry args={[0.8, 0.8, 2, 16]} />
            <meshStandardMaterial color="#c0c0c0" metalness={0.8} roughness={0.2} />
          </mesh>
          
          {/* Solar Panel Left */}
          <mesh ref={leftPanelRef} position={[0, 0, 0]}>
            <boxGeometry args={[1.5, 3, 0.1]} />
            <meshStandardMaterial color="#1a3b5c" metalness={0.5} />
            <gridHelper args={[1.5, 8, 0x000000, 0x000000]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.06]} />
          </mesh>

          {/* Solar Panel Right */}
          <mesh ref={rightPanelRef} position={[0, 0, 0]}>
            <boxGeometry args={[1.5, 3, 0.1]} />
            <meshStandardMaterial color="#1a3b5c" metalness={0.5} />
            <gridHelper args={[1.5, 8, 0x000000, 0x000000]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.06]} />
          </mesh>

          {/* Dish */}
          <mesh position={[0, 1.2, 0]}>
            <cylinderGeometry args={[1, 0.1, 0.5, 16]} />
            <meshStandardMaterial color="#ffffff" metalness={0.3} />
          </mesh>

          <Html distanceFactor={10} position={[0, 2, 0]}>
            <div ref={htmlRef} className="text-[#00ffcc] font-mono text-sm bg-black/80 px-3 py-2 border border-[#00ffcc]/40 rounded whitespace-nowrap opacity-0 transition-opacity">
              &gt; SATELLITE DEPLOYED<br/>
              &gt; ORBIT ACHIEVED
            </div>
          </Html>
        </group>

      </group>
    </group>
  );
}

export default function RocketLaunchSequence() {
  return (
    <div className="w-full h-[100dvh] bg-[#000005] relative flex flex-col">
      <div className="absolute top-8 w-full z-20 text-center pointer-events-none px-4">
        <h2 className="text-white text-xl md:text-3xl font-mono tracking-[0.2em] font-bold uppercase drop-shadow-md">
          Mission Sequence
        </h2>
        <p className="text-white/60 text-xs mt-2 font-mono uppercase tracking-widest">
          Scroll down to launch
        </p>
      </div>

      <Canvas camera={{ position: [0, 2, 10], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 10]} intensity={1.5} />
        
        <Stars radius={50} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        
        <ScrollControls pages={4} damping={0.2}>
          <LaunchAnimation />
        </ScrollControls>
      </Canvas>
      
      <div className="absolute bottom-6 w-full z-20 flex justify-center pointer-events-none">
        <div className="w-[1px] h-16 bg-gradient-to-b from-white/50 to-transparent"></div>
      </div>
    </div>
  );
}
