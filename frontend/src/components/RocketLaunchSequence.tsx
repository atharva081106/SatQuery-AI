"use client";

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ScrollControls, useScroll, Html, Stars, useTexture, Environment } from '@react-three/drei';
import { EffectComposer, Bloom, ToneMapping } from '@react-three/postprocessing';
import * as THREE from 'three';

function LaunchAnimation() {
  const scroll = useScroll();
  
  // References
  const earthRef = useRef<THREE.Mesh>(null);
  const cloudsRef = useRef<THREE.Mesh>(null);
  const atmosphereRef = useRef<THREE.Mesh>(null);
  
  const rocketGroupRef = useRef<THREE.Group>(null);
  const exhaustRef = useRef<THREE.Mesh>(null);
  const exhaustLightRef = useRef<THREE.PointLight>(null);
  
  const fairingLeftRef = useRef<THREE.Group>(null);
  const fairingRightRef = useRef<THREE.Group>(null);
  
  const satelliteGroupRef = useRef<THREE.Group>(null);
  const leftPanelRef = useRef<THREE.Mesh>(null);
  const rightPanelRef = useRef<THREE.Mesh>(null);
  
  const htmlRef = useRef<HTMLDivElement>(null);

  const earthTexture = useTexture('/textures/2k_earth_daymap.jpg');
  const cloudsTexture = useTexture('/textures/2k_earth_clouds.jpg');

  useFrame((state, delta) => {
    const offset = scroll.offset; // 0 to 1
    
    // Rotate Earth slowly
    if (earthRef.current) earthRef.current.rotation.y += 0.05 * delta;
    if (cloudsRef.current) cloudsRef.current.rotation.y += 0.06 * delta; // clouds move slightly faster

    // CAMERA LOGIC
    if (offset < 0.2) {
      state.camera.position.lerp(new THREE.Vector3(0, 2, 14), 0.05);
      state.camera.lookAt(0, 2, 0);
    } else if (offset < 0.5) {
      const t = (offset - 0.2) / 0.3;
      state.camera.position.lerp(new THREE.Vector3(0, 2 - t * 4, 14 + t * 4), 0.05);
      state.camera.lookAt(0, 0, 0);
    } else {
      state.camera.position.lerp(new THREE.Vector3(0, 4, 8), 0.05);
      state.camera.lookAt(0, 4, 0);
    }

    // ANIMATION STAGES

    // Stage 1: Liftoff & Earth Descent (offset 0.1 to 0.45)
    const liftoffProgress = Math.max(0, Math.min(1, (offset - 0.1) / 0.35));
    
    if (earthRef.current && atmosphereRef.current && cloudsRef.current) {
      const targetScale = 1 - (liftoffProgress * 0.95);
      earthRef.current.scale.set(targetScale, targetScale, targetScale);
      cloudsRef.current.scale.set(targetScale * 1.01, targetScale * 1.01, targetScale * 1.01);
      atmosphereRef.current.scale.set(targetScale * 1.03, targetScale * 1.03, targetScale * 1.03);
      
      const dropY = -20 - (liftoffProgress * 60);
      earthRef.current.position.y = dropY;
      cloudsRef.current.position.y = dropY;
      atmosphereRef.current.position.y = dropY;
    }
    
    if (exhaustRef.current && exhaustLightRef.current) {
      if (liftoffProgress > 0 && liftoffProgress < 1) {
        exhaustRef.current.scale.set(1 + Math.random() * 0.2, 1 + Math.random() * 0.8, 1 + Math.random() * 0.2);
        exhaustRef.current.visible = true;
        // Flicker intensity for realism
        exhaustLightRef.current.intensity = 15 + Math.random() * 10;
        (exhaustRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 5 + Math.random() * 5;
      } else {
        exhaustRef.current.visible = false;
        exhaustLightRef.current.intensity = 0;
      }
    }
    
    if (rocketGroupRef.current) {
      if (liftoffProgress > 0 && liftoffProgress < 1) {
        const shake = (1 - liftoffProgress) * 0.08;
        rocketGroupRef.current.position.x = (Math.random() - 0.5) * shake;
        rocketGroupRef.current.position.z = (Math.random() - 0.5) * shake;
      } else {
        rocketGroupRef.current.position.x = 0;
        rocketGroupRef.current.position.z = 0;
      }
    }

    // Stage 2: Fairing Separation
    const fairingProgress = Math.max(0, Math.min(1, (offset - 0.5) / 0.2));
    
    if (fairingLeftRef.current) {
      fairingLeftRef.current.position.x = -fairingProgress * 8;
      fairingLeftRef.current.rotation.z = fairingProgress * 1.5;
    }
    if (fairingRightRef.current) {
      fairingRightRef.current.position.x = fairingProgress * 8;
      fairingRightRef.current.rotation.z = -fairingProgress * 1.5;
    }

    // Stage 3: Satellite Deployment
    const deployProgress = Math.max(0, Math.min(1, (offset - 0.7) / 0.25));
    
    if (satelliteGroupRef.current) {
      satelliteGroupRef.current.position.y = 2.5 + (deployProgress * 2.5);
      if (deployProgress > 0) {
        satelliteGroupRef.current.rotation.y += 0.8 * delta;
        satelliteGroupRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      }
    }
    
    if (leftPanelRef.current) {
      leftPanelRef.current.position.x = THREE.MathUtils.lerp(0, -1.8, deployProgress);
      leftPanelRef.current.rotation.y = THREE.MathUtils.lerp(Math.PI/2, 0, deployProgress);
    }
    if (rightPanelRef.current) {
      rightPanelRef.current.position.x = THREE.MathUtils.lerp(0, 1.8, deployProgress);
      rightPanelRef.current.rotation.y = THREE.MathUtils.lerp(-Math.PI/2, 0, deployProgress);
    }

    // HTML HUD
    if (htmlRef.current) {
      const opacity = offset > 0.95 ? (offset - 0.95) * 20 : 0;
      htmlRef.current.style.opacity = Math.min(1, opacity).toString();
      htmlRef.current.style.transform = `translateY(${Math.max(0, 20 - opacity * 20)}px)`;
    }
  });

  // Physically Based Materials (PBR)
  const rocketMaterial = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: '#ffffff', 
    metalness: 0.7, 
    roughness: 0.2,
    envMapIntensity: 2.0 // highly reflective to environment
  }), []);

  const darkMetal = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: '#222222', 
    metalness: 0.9, 
    roughness: 0.3,
    envMapIntensity: 1.5
  }), []);

  const goldFoil = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: '#ffcc00', 
    metalness: 0.8, 
    roughness: 0.4, 
    envMapIntensity: 1.5
  }), []);

  return (
    <group>
      {/* EARTH */}
      <mesh ref={earthRef} position={[0, -20, 0]}>
        <sphereGeometry args={[18, 64, 64]} />
        <meshStandardMaterial map={earthTexture} roughness={0.7} metalness={0.1} />
      </mesh>
      
      {/* CLOUDS LAYER (Realistic volumetric look) */}
      <mesh ref={cloudsRef} position={[0, -20, 0]}>
        <sphereGeometry args={[18.05, 64, 64]} />
        <meshStandardMaterial map={cloudsTexture} transparent opacity={0.6} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>

      {/* ATMOSPHERE GLOW */}
      <mesh ref={atmosphereRef} position={[0, -20, 0]}>
        <sphereGeometry args={[18.4, 64, 64]} />
        <meshBasicMaterial color="#4b70dd" transparent opacity={0.15} side={THREE.BackSide} />
      </mesh>

      {/* ROCKET OVERALL GROUP */}
      <group ref={rocketGroupRef} position={[0, -1, 0]}>
        
        {/* ROCKET STAGE 1 (Booster) */}
        <mesh position={[0, -0.5, 0]}>
          <cylinderGeometry args={[0.6, 0.6, 5, 64]} />
          <primitive object={rocketMaterial} attach="material" />
        </mesh>

        {/* ENGINE BELL */}
        <mesh position={[0, -3.2, 0]}>
          <cylinderGeometry args={[0.4, 0.7, 0.6, 32]} />
          <primitive object={darkMetal} attach="material" />
        </mesh>
        
        {/* EXHAUST FLAME (Uses bloom postprocessing) */}
        <mesh ref={exhaustRef} position={[0, -4.5, 0]} visible={false}>
          <coneGeometry args={[0.5, 2.5, 32]} />
          <meshStandardMaterial color="#ff5500" emissive="#ffaa00" emissiveIntensity={5} transparent opacity={0.9} />
          <pointLight ref={exhaustLightRef} color="#ff7700" distance={30} decay={1.5} intensity={0} />
        </mesh>

        {/* FINS */}
        <group position={[0, -2.5, 0]}>
          {[0, Math.PI/2, Math.PI, Math.PI*1.5].map((rot, i) => (
            <mesh key={i} rotation={[0, rot, 0]} position={[Math.sin(rot)*0.7, 0, Math.cos(rot)*0.7]}>
              <boxGeometry args={[0.1, 1.2, 0.8]} />
              <primitive object={rocketMaterial} attach="material" />
            </mesh>
          ))}
        </group>

        {/* FAIRING LEFT */}
        <group ref={fairingLeftRef} position={[0, 2, 0]}>
          <mesh position={[-0.3, 0.5, 0]}>
            <cylinderGeometry args={[0.6, 0.6, 2, 32, 1, false, Math.PI, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
          <mesh position={[-0.3, 2, 0]}>
            <coneGeometry args={[0.6, 1.5, 32, 1, false, Math.PI, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
        </group>

        {/* FAIRING RIGHT */}
        <group ref={fairingRightRef} position={[0, 2, 0]}>
          <mesh position={[0.3, 0.5, 0]}>
            <cylinderGeometry args={[0.6, 0.6, 2, 32, 1, false, 0, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
          <mesh position={[0.3, 2, 0]}>
            <coneGeometry args={[0.6, 1.5, 32, 1, false, 0, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
        </group>

        {/* PAYLOAD (Satellite) */}
        <group ref={satelliteGroupRef} position={[0, 2.5, 0]} scale={[0.5, 0.5, 0.5]}>
          {/* Main Bus (Gold Foil) */}
          <mesh>
            <boxGeometry args={[1.2, 1.5, 1.2]} />
            <primitive object={goldFoil} attach="material" />
          </mesh>
          
          {/* Solar Panel Left */}
          <mesh ref={leftPanelRef} position={[0, 0, 0]}>
            <boxGeometry args={[2, 4, 0.05]} />
            <meshStandardMaterial color="#0a1b2c" metalness={1} roughness={0.2} envMapIntensity={2} />
            <gridHelper args={[2, 10, 0x4488ff, 0x4488ff]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.03]} />
          </mesh>

          {/* Solar Panel Right */}
          <mesh ref={rightPanelRef} position={[0, 0, 0]}>
            <boxGeometry args={[2, 4, 0.05]} />
            <meshStandardMaterial color="#0a1b2c" metalness={1} roughness={0.2} envMapIntensity={2} />
            <gridHelper args={[2, 10, 0x4488ff, 0x4488ff]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.03]} />
          </mesh>

          {/* Dish */}
          <mesh position={[0, 1.0, 0]} rotation={[-0.2, 0, 0]}>
            <cylinderGeometry args={[0.8, 0.1, 0.4, 32]} />
            <meshStandardMaterial color="#ffffff" metalness={0.5} roughness={0.5} />
          </mesh>
          
          {/* Antenna mast */}
          <mesh position={[0, 1.4, 0]}>
            <cylinderGeometry args={[0.05, 0.05, 1, 16]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>

          <Html distanceFactor={10} position={[0, 2.5, 0]} center>
            <div ref={htmlRef} className="flex flex-col items-center opacity-0 transition-all duration-500 pointer-events-none">
              <div className="text-[#00ffcc] font-mono text-lg md:text-2xl tracking-widest font-bold bg-black/60 px-4 py-2 border border-[#00ffcc]/50 rounded-sm shadow-[0_0_15px_rgba(0,255,204,0.5)] backdrop-blur-sm text-center">
                ORBIT ACHIEVED
              </div>
              <div className="w-[1px] h-8 bg-gradient-to-b from-[#00ffcc]/80 to-transparent mt-2"></div>
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
      <div className="absolute top-24 w-full z-20 text-center pointer-events-none px-4">
        <h2 className="text-white text-2xl md:text-4xl font-mono tracking-[0.3em] font-bold uppercase drop-shadow-[0_0_10px_rgba(255,255,255,0.5)]">
          Launch Sequence
        </h2>
        <p className="text-[#00ffcc] text-xs mt-3 font-mono uppercase tracking-[0.2em] animate-pulse">
          Scroll Down to Initiate
        </p>
      </div>

      <Canvas camera={{ position: [0, 2, 12], fov: 45 }}>
        {/* Photorealistic Lighting via HDRI Environment */}
        <Environment preset="night" background={false} />
        
        <ambientLight intensity={0.2} />
        <directionalLight position={[10, 10, 10]} intensity={2.5} color="#ffeedd" castShadow />
        
        <Stars radius={100} depth={50} count={8000} factor={4} saturation={0} fade speed={1.5} />
        
        <ScrollControls pages={5} damping={0.2}>
          <LaunchAnimation />
        </ScrollControls>

        {/* Cinematic Post-Processing */}
        <EffectComposer disableNormalPass>
          <Bloom luminanceThreshold={1} mipmapBlur intensity={1.5} />
          <ToneMapping />
        </EffectComposer>
      </Canvas>
      
      <div className="absolute bottom-10 w-full z-20 flex flex-col items-center justify-center pointer-events-none">
        <div className="w-[2px] h-12 bg-gradient-to-b from-white/80 to-transparent animate-bounce"></div>
      </div>
    </div>
  );
}
