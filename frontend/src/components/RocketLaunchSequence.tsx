"use client";

import React, { useRef, useMemo, useState, useEffect, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ScrollControls, useScroll, Stars, useTexture } from '@react-three/drei';
import { EffectComposer, Bloom, Noise, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';

// Utility to create a timeline progress between two offsets [start, end]
function getProgress(offset: number, start: number, end: number) {
  return Math.max(0, Math.min(1, (offset - start) / (end - start)));
}

// Cubic ease-out for smooth natural deceleration
function easeOut(t: number) {
  return 1 - Math.pow(1 - t, 3);
}

// Stage configuration for interactive milestones
const STAGE_MILESTONES = [
  { id: 'liftoff', label: '01. LIFTOFF', offset: 0.0, met: 'T+ 00:00:00', title: 'PAD T-0 // BOOSTER IGNITION & LIFTOFF' },
  { id: 'maxq', label: '02. MAX-Q', offset: 0.22, met: 'T+ 00:01:12', title: 'MAX-Q // PEAK DYNAMIC PRESSURE' },
  { id: 'stage1', label: '03. STAGE 1 SEP', offset: 0.34, met: 'T+ 00:01:54', title: 'MECO // STAGE 1 BOOSTER JETTISON' },
  { id: 'fairing', label: '04. FAIRING SEP', offset: 0.48, met: 'T+ 00:02:40', title: 'PAYLOAD FAIRING JETTISON // EXPOSURE' },
  { id: 'stage2', label: '05. STAGE 2 SEP', offset: 0.72, met: 'T+ 00:04:30', title: 'SECO // STAGE 2 CUTOFF & PAYLOAD SEP' },
  { id: 'orbit', label: '06. ORBIT INSERTION', offset: 0.95, met: 'T+ 00:10:15', title: 'ORBIT ACHIEVED // SOLAR ARRAYS DEPLOYED' }
];

// Helper to format telemetry based on scroll offset
function computeTelemetry(offset: number) {
  // Altitude: 0 -> 525 km (non-linear ascent curve)
  let altitude = 0;
  if (offset < 0.2) {
    altitude = offset * 5 * 25; // 0 to 25 km
  } else if (offset < 0.7) {
    altitude = 25 + (offset - 0.2) * 2 * 350; // 25 to 375 km
  } else {
    altitude = 375 + (offset - 0.7) * 3.33 * 150; // 375 to 525 km
  }

  // Velocity: 0 -> 7.68 km/s
  let velocity = 0;
  if (offset < 0.3) {
    velocity = (offset / 0.3) * 2.3;
  } else if (offset < 0.7) {
    velocity = 2.3 + ((offset - 0.3) / 0.4) * 4.9;
  } else {
    velocity = 7.2 + ((offset - 0.7) / 0.3) * 0.48;
  }

  // Time in seconds: 0 -> 615 seconds
  const totalSeconds = Math.floor(offset * 615);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const met = `T+ ${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

  // Stage description
  let stageName = STAGE_MILESTONES[0].title;
  let activeIndex = 0;
  for (let i = 0; i < STAGE_MILESTONES.length; i++) {
    if (offset >= STAGE_MILESTONES[i].offset - 0.05) {
      stageName = STAGE_MILESTONES[i].title;
      activeIndex = i;
    }
  }

  return {
    altitude: altitude.toFixed(1),
    velocity: velocity.toFixed(2),
    met,
    stageName,
    activeIndex,
    progress: (offset * 100).toFixed(1)
  };
}

// Controller inside ScrollControls that accesses scroll state & runs 60fps animations
function LaunchController({ isAutoPlaying, setIsAutoPlaying }: { isAutoPlaying: boolean; setIsAutoPlaying: (v: boolean) => void }) {
  const scroll = useScroll();
  
  // Camera smoothing ref
  const currentLookAt = useRef(new THREE.Vector3(0, 6, 0));
  const continuousAngleRef = useRef(0);
  
  // Scenery Refs
  const earthGroupRef = useRef<THREE.Group>(null);
  const cloudsRef = useRef<THREE.Mesh>(null);
  
  // Wrappers (trajectory)
  const stage1WrapperRef = useRef<THREE.Group>(null);
  const stage2WrapperRef = useRef<THREE.Group>(null);
  const fairingLeftWrapperRef = useRef<THREE.Group>(null);
  const fairingRightWrapperRef = useRef<THREE.Group>(null);
  const satelliteWrapperRef = useRef<THREE.Group>(null);
  
  // Components
  const stage1GroupRef = useRef<THREE.Group>(null);
  const exhaust1Ref = useRef<THREE.Mesh>(null);
  const light1Ref = useRef<THREE.PointLight>(null);
  
  const stage2GroupRef = useRef<THREE.Group>(null);
  const exhaust2Ref = useRef<THREE.Mesh>(null);
  const light2Ref = useRef<THREE.PointLight>(null);
  
  const fairingLeftRef = useRef<THREE.Group>(null);
  const fairingRightRef = useRef<THREE.Group>(null);
  
  const satelliteGroupRef = useRef<THREE.Group>(null);
  const leftPanelRef = useRef<THREE.Mesh>(null);
  const rightPanelRef = useRef<THREE.Mesh>(null);

  // Load textures
  const earthTexture = useTexture('/textures/2k_earth_daymap.jpg');
  const cloudsTexture = useTexture('/textures/2k_earth_clouds.jpg');

  // Shared Materials (Optimized: single instances, no primitive re-wrapping)
  const rocketMaterial = useMemo(() => new THREE.MeshPhysicalMaterial({ 
    color: '#f0f0f0', 
    metalness: 0.7, 
    roughness: 0.2, 
    clearcoat: 0.8, 
    clearcoatRoughness: 0.15 
  }), []);

  const darkMetal = useMemo(() => new THREE.MeshPhysicalMaterial({ 
    color: '#181818', 
    metalness: 0.9, 
    roughness: 0.35 
  }), []);

  const goldFoil = useMemo(() => new THREE.MeshPhysicalMaterial({ 
    color: '#ffb703', 
    metalness: 0.95, 
    roughness: 0.25, 
    clearcoat: 0.6 
  }), []);

  const engineFlame = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: '#fff', 
    emissive: '#ff7700', 
    emissiveIntensity: 12, 
    transparent: true, 
    opacity: 0.9 
  }), []);

  const vacFlame = useMemo(() => new THREE.MeshStandardMaterial({ 
    color: '#fff', 
    emissive: '#00b4d8', 
    emissiveIntensity: 8, 
    transparent: true, 
    opacity: 0.85 
  }), []);

  // Shared Ring Torus Geometry for stage bands
  const ringGeo = useMemo(() => new THREE.TorusGeometry(0.6, 0.025, 12, 32), []);

  // Refs for tracking DOM updates to prevent layout thrashing
  const lastState = useRef({
    met: '', alt: '', vel: '', stageName: '', pct: '', activeIndex: -1
  });

  const freeOrbitRef = useRef(0);

  // Trajectory function - fully deterministic for perfect bidirectional scrolling
  const getTrajectory = (clampedOffset: number) => {
    const pLiftoff = getProgress(clampedOffset, 0.0, 0.2);
    const pOrbit = getProgress(clampedOffset, 0.2, 0.7);
    const pPitch = getProgress(clampedOffset, 0.1, 0.2);
    
    const orbitRadius = 25;
    let orbitAngle = 0;
    let posX = 0;
    let posY = -1 + (pLiftoff * 6);
    let posZ = 0;
    
    let rotX = 0;
    let rotY = 0;
    let rotZ = THREE.MathUtils.lerp(0, -(Math.PI / 2), pPitch);

    if (pOrbit > 0) {
      // 0.7 to 1.0 adds additional orbit rotation based purely on scroll
      const additionalOrbit = getProgress(clampedOffset, 0.7, 1.0) * Math.PI * 0.9;
      orbitAngle = (pOrbit * (Math.PI / 2)) + additionalOrbit + freeOrbitRef.current;
      posX = Math.sin(orbitAngle) * orbitRadius;
      posY = -20 + Math.cos(orbitAngle) * orbitRadius;
      rotZ = THREE.MathUtils.lerp(0, -(Math.PI / 2), pPitch) - orbitAngle;
    }

    // Max-Q subtle vibration
    const pMaxQ = getProgress(clampedOffset, 0.18, 0.26);
    if (pMaxQ > 0 && pMaxQ < 1) {
      posX += (Math.random() - 0.5) * 0.04;
    }

    return { posX, posY, posZ, rotZ, rotY, rotX };
  };

  // Expose jumpToStage on window so UI buttons can trigger smooth jump
  useEffect(() => {
    const handleJump = (e: CustomEvent<{ offset: number }>) => {
      if (scroll && scroll.el) {
        const maxScroll = scroll.el.scrollHeight - scroll.el.clientHeight;
        scroll.el.scrollTo({ top: e.detail.offset * maxScroll, behavior: 'smooth' });
      }
    };
    window.addEventListener('jump-to-stage' as any, handleJump);
    return () => window.removeEventListener('jump-to-stage' as any, handleJump);
  }, [scroll]);

  // Frame Loop
  useFrame((state, delta) => {
    // Handle Auto-Play
    if (isAutoPlaying && scroll && scroll.el) {
      const maxScroll = scroll.el.scrollHeight - scroll.el.clientHeight;
      scroll.el.scrollTop += delta * maxScroll * 0.045; // Slower cinematic playback
      if (scroll.el.scrollTop >= maxScroll - 2) {
        setIsAutoPlaying(false);
      }
    }

    const offset = scroll.offset;

    if (offset > 0.99) {
      freeOrbitRef.current += delta * 0.12; // Perpetual orbit drift at 100%
    } else if (freeOrbitRef.current !== 0) {
      // Normalize to [-PI, PI] to prevent snapping when scrolling back
      let diff = freeOrbitRef.current % (Math.PI * 2);
      if (diff > Math.PI) diff -= Math.PI * 2;
      if (diff < -Math.PI) diff += Math.PI * 2;
      
      // Smoothly unwind back to zero, ensuring perfect return to deterministic scroll
      freeOrbitRef.current = THREE.MathUtils.lerp(diff, 0, 0.1);
      if (Math.abs(freeOrbitRef.current) < 0.001) freeOrbitRef.current = 0;
    }

    // Direct HUD DOM updates with cache to prevent style recalculation thrashing
    const tel = computeTelemetry(offset);
    
    if (tel.met !== lastState.current.met) {
      const el = document.getElementById('hud-met');
      if (el) el.textContent = tel.met;
      lastState.current.met = tel.met;
    }
    if (tel.altitude !== lastState.current.alt) {
      const el = document.getElementById('hud-altitude');
      if (el) el.textContent = `${tel.altitude} KM`;
      lastState.current.alt = tel.altitude;
    }
    if (tel.velocity !== lastState.current.vel) {
      const el = document.getElementById('hud-velocity');
      if (el) el.textContent = `${tel.velocity} KM/S`;
      lastState.current.vel = tel.velocity;
    }
    if (tel.stageName !== lastState.current.stageName) {
      const el = document.getElementById('hud-status');
      if (el) el.textContent = tel.stageName;
      lastState.current.stageName = tel.stageName;
    }
    if (tel.progress !== lastState.current.pct) {
      const elBar = document.getElementById('hud-progress-bar');
      const elPct = document.getElementById('hud-percent');
      if (elBar) elBar.style.width = `${tel.progress}%`;
      if (elPct) elPct.textContent = `${tel.progress}%`;
      lastState.current.pct = tel.progress;
    }

    // Highlight active milestone pill only if changed
    if (tel.activeIndex !== lastState.current.activeIndex) {
      lastState.current.activeIndex = tel.activeIndex;
      for (let i = 0; i < STAGE_MILESTONES.length; i++) {
        const pill = document.getElementById(`milestone-pill-${i}`);
        if (pill) {
          if (i === tel.activeIndex) {
            pill.className = "px-2.5 py-1 rounded font-mono text-[10px] tracking-wider uppercase bg-white text-black font-bold shadow-lg shadow-white/20 transition-all scale-105";
          } else if (i < tel.activeIndex) {
            pill.className = "px-2.5 py-1 rounded font-mono text-[10px] tracking-wider uppercase bg-white/10 text-white/80 hover:bg-white/20 transition-all";
          } else {
            pill.className = "px-2.5 py-1 rounded font-mono text-[10px] tracking-wider uppercase bg-black/40 text-white/40 hover:text-white/70 transition-all border border-white/5";
          }
        }
      }
    }

    // Earth & Cloud gentle axial rotation
    if (earthGroupRef.current) earthGroupRef.current.rotation.y += 0.015 * delta;
    if (cloudsRef.current) cloudsRef.current.rotation.y += 0.022 * delta;

    // Trajectory coordinates for payload
    const satTraj = getTrajectory(offset);
    if (satelliteWrapperRef.current) {
      satelliteWrapperRef.current.position.set(satTraj.posX, satTraj.posY, satTraj.posZ);
      satelliteWrapperRef.current.rotation.set(satTraj.rotX, satTraj.rotY, satTraj.rotZ, 'YXZ');
    }

    const pSatSep = getProgress(offset, 0.70, 0.76);
    const pSatAcq = getProgress(offset, 0.80, 0.92);
    
    if (satelliteGroupRef.current) {
      satelliteGroupRef.current.position.y = pSatSep * 3;
      if (pSatSep > 0) {
        satelliteGroupRef.current.rotation.y += 0.4 * delta;
        satelliteGroupRef.current.rotation.x = THREE.MathUtils.lerp(0, Math.PI / 4, pSatAcq);
      }
    }

    // --- STAGE 1 (Booster) ---
    const s1Offset = Math.min(offset, 0.30);
    const s1Traj = getTrajectory(s1Offset);
    if (stage1WrapperRef.current) {
      stage1WrapperRef.current.position.set(s1Traj.posX, s1Traj.posY, s1Traj.posZ);
      stage1WrapperRef.current.rotation.set(s1Traj.rotX, s1Traj.rotY, s1Traj.rotZ, 'YXZ');
    }

    const pStage1SepRaw = getProgress(offset, 0.30, 0.58);
    const pStage1Sep = easeOut(pStage1SepRaw);
    if (stage1GroupRef.current) {
      stage1GroupRef.current.position.y = -pStage1Sep * 14;
      stage1GroupRef.current.position.x = -pStage1Sep * 2.5;
      stage1GroupRef.current.rotation.z = pStage1Sep * 3.5;
      const scale = Math.max(0, 1 - (pStage1SepRaw * 1.3));
      stage1GroupRef.current.scale.setScalar(scale);
    }

    // --- FAIRINGS ---
    const fairingOffset = Math.min(offset, 0.42);
    const fairingTraj = getTrajectory(fairingOffset);
    if (fairingLeftWrapperRef.current && fairingRightWrapperRef.current) {
      fairingLeftWrapperRef.current.position.set(fairingTraj.posX, fairingTraj.posY, fairingTraj.posZ);
      fairingLeftWrapperRef.current.rotation.set(fairingTraj.rotX, fairingTraj.rotY, fairingTraj.rotZ, 'YXZ');
      fairingRightWrapperRef.current.position.set(fairingTraj.posX, fairingTraj.posY, fairingTraj.posZ);
      fairingRightWrapperRef.current.rotation.set(fairingTraj.rotX, fairingTraj.rotY, fairingTraj.rotZ, 'YXZ');
    }

    const pFairingSepRaw = getProgress(offset, 0.42, 0.65);
    const pFairingSep = easeOut(pFairingSepRaw);
    if (fairingLeftRef.current && fairingRightRef.current) {
      fairingLeftRef.current.position.x = -pFairingSep * 5;
      fairingLeftRef.current.position.y = -pFairingSep * 9;
      fairingLeftRef.current.rotation.z = pFairingSep * 3.5;

      fairingRightRef.current.position.x = pFairingSep * 5;
      fairingRightRef.current.position.y = -pFairingSep * 9;
      fairingRightRef.current.rotation.z = -pFairingSep * 3.5;

      const fScale = Math.max(0, 1 - (pFairingSepRaw * 1.5));
      fairingLeftRef.current.scale.setScalar(fScale);
      fairingRightRef.current.scale.setScalar(fScale);
    }

    // --- STAGE 2 (Upper Stage) ---
    const s2Offset = Math.min(offset, 0.70);
    const s2Traj = getTrajectory(s2Offset);
    if (stage2WrapperRef.current) {
      stage2WrapperRef.current.position.set(s2Traj.posX, s2Traj.posY, s2Traj.posZ);
      stage2WrapperRef.current.rotation.set(s2Traj.rotX, s2Traj.rotY, s2Traj.rotZ, 'YXZ');
    }

    const pStage2SepRaw = getProgress(offset, 0.70, 0.88);
    const pStage2Sep = easeOut(pStage2SepRaw);
    if (stage2GroupRef.current) {
      stage2GroupRef.current.position.y = -pStage2Sep * 14;
      stage2GroupRef.current.rotation.z = pStage2Sep * 2;
      const s2Scale = Math.max(0, 1 - (pStage2SepRaw * 1.5));
      stage2GroupRef.current.scale.setScalar(s2Scale);
    }

    // --- CAMERA LOGIC ---
    const targetCamPos = new THREE.Vector3();
    const targetLookAt = new THREE.Vector3();

    if (satelliteWrapperRef.current && satelliteGroupRef.current) {
      const vPos = new THREE.Vector3();
      satelliteGroupRef.current.getWorldPosition(vPos);
      
      const aspect = state.size.width / state.size.height;
      const isMobile = aspect < 1;

      if (offset < 0.9) {
        const launchZoom = isMobile ? 60 : 42;
        targetCamPos.set(vPos.x, vPos.y + 5, launchZoom);
        targetLookAt.set(vPos.x, vPos.y + 5, 0);
      } else {
        const finalZoom = isMobile ? 100 : 75;
        targetCamPos.set(0, -20, finalZoom);
        targetLookAt.set(0, -20, 0);
      }
    }

    const lerpFactor = offset < 0.9 ? 0.18 : 0.04;
    state.camera.position.lerp(targetCamPos, lerpFactor);
    currentLookAt.current.lerp(targetLookAt, lerpFactor);
    state.camera.lookAt(currentLookAt.current);

    // --- EXHAUST 1 (Booster) ---
    const pStage1Burnout = getProgress(offset, 0.28, 0.30);
    if (exhaust1Ref.current && light1Ref.current) {
      if (offset > 0.01 && pStage1Burnout < 1) {
        exhaust1Ref.current.visible = true;
        const intensity = 1 - pStage1Burnout;
        exhaust1Ref.current.scale.set(1 + Math.random() * 0.15, (1 + Math.random() * 0.6) * intensity, 1 + Math.random() * 0.15);
        light1Ref.current.intensity = (14 + Math.random() * 8) * intensity;
      } else {
        exhaust1Ref.current.visible = false;
        light1Ref.current.intensity = 0;
      }
    }

    // --- EXHAUST 2 (Vacuum Engine) ---
    const pStage2Ignition = getProgress(offset, 0.34, 0.40);
    const pStage2Cutoff = getProgress(offset, 0.66, 0.70);
    if (exhaust2Ref.current && light2Ref.current) {
      if (pStage2Ignition > 0 && pStage2Cutoff < 1) {
        exhaust2Ref.current.visible = true;
        const intensity = pStage2Ignition * (1 - pStage2Cutoff);
        exhaust2Ref.current.scale.set((1.3 + Math.random() * 0.2) * intensity, (0.8 + Math.random() * 0.3) * intensity, (1.3 + Math.random() * 0.2) * intensity);
        light2Ref.current.intensity = (6 + Math.random() * 4) * intensity;
      } else {
        exhaust2Ref.current.visible = false;
        light2Ref.current.intensity = 0;
      }
    }

    // --- SOLAR ARRAYS ---
    if (leftPanelRef.current && rightPanelRef.current) {
      leftPanelRef.current.position.x = THREE.MathUtils.lerp(0, -1.8, pSatAcq);
      leftPanelRef.current.rotation.y = THREE.MathUtils.lerp(Math.PI / 2, 0, pSatAcq);
      
      rightPanelRef.current.position.x = THREE.MathUtils.lerp(0, 1.8, pSatAcq);
      rightPanelRef.current.rotation.y = THREE.MathUtils.lerp(-Math.PI / 2, 0, pSatAcq);
    }
  });

  return (
    <group>
      {/* EARTH SYSTEM (Optimized to 48 segments for high performance) */}
      <group ref={earthGroupRef} position={[0, -20, 0]}>
        <mesh>
          <sphereGeometry args={[18, 48, 48]} />
          <meshStandardMaterial map={earthTexture} roughness={0.85} metalness={0.1} />
        </mesh>
        <mesh ref={cloudsRef}>
          <sphereGeometry args={[18.08, 48, 48]} />
          <meshStandardMaterial map={cloudsTexture} transparent opacity={0.45} depthWrite={false} blending={THREE.AdditiveBlending} />
        </mesh>
        {/* Sleek Atmospheric Glow */}
        <mesh>
          <sphereGeometry args={[18.35, 36, 36]} />
          <meshBasicMaterial color="#3a86ff" transparent opacity={0.18} side={THREE.BackSide} blending={THREE.AdditiveBlending} />
        </mesh>
      </group>

      {/* SATELLITE (Payload) */}
      <group ref={satelliteWrapperRef}>
        <group ref={satelliteGroupRef}>
          <group position={[0, 10, 0]} scale={[0.5, 0.5, 0.5]}>
            <mesh material={goldFoil}>
              <boxGeometry args={[1.2, 2.5, 1.2]} />
            </mesh>
            
            <mesh ref={leftPanelRef} position={[0, 0, 0]}>
              <boxGeometry args={[2, 4, 0.05]} />
              <meshStandardMaterial color="#0c1b33" metalness={0.9} roughness={0.2} />
              <gridHelper args={[2, 8, 0x4488ff, 0x3366cc]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.03]} />
            </mesh>

            <mesh ref={rightPanelRef} position={[0, 0, 0]}>
              <boxGeometry args={[2, 4, 0.05]} />
              <meshStandardMaterial color="#0c1b33" metalness={0.9} roughness={0.2} />
              <gridHelper args={[2, 8, 0x4488ff, 0x3366cc]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.03]} />
            </mesh>

            <mesh position={[0, 1.5, 0]} rotation={[-0.2, 0, 0]}>
              <cylinderGeometry args={[0.8, 0.1, 0.4, 24]} />
              <meshStandardMaterial color="#ffffff" metalness={0.6} roughness={0.4} />
            </mesh>
            
            <mesh position={[0, 1.9, 0]} material={darkMetal}>
              <cylinderGeometry args={[0.05, 0.05, 1, 12]} />
            </mesh>
          </group>
        </group>
      </group>

      {/* FAIRING LEFT */}
      <group ref={fairingLeftWrapperRef}>
        <group ref={fairingLeftRef}>
          <mesh position={[0, 10, 0]} material={rocketMaterial}>
            <cylinderGeometry args={[0.62, 0.62, 2, 24, 1, false, Math.PI, Math.PI]} />
          </mesh>
          <mesh position={[0, 11.75, 0]} material={rocketMaterial}>
            <coneGeometry args={[0.62, 1.5, 24, 1, false, Math.PI, Math.PI]} />
          </mesh>
        </group>
      </group>

      {/* FAIRING RIGHT */}
      <group ref={fairingRightWrapperRef}>
        <group ref={fairingRightRef}>
          <mesh position={[0, 10, 0]} material={rocketMaterial}>
            <cylinderGeometry args={[0.62, 0.62, 2, 24, 1, false, 0, Math.PI]} />
          </mesh>
          <mesh position={[0, 11.75, 0]} material={rocketMaterial}>
            <coneGeometry args={[0.62, 1.5, 24, 1, false, 0, Math.PI]} />
          </mesh>
        </group>
      </group>

      {/* STAGE 2 */}
      <group ref={stage2WrapperRef}>
        <group ref={stage2GroupRef}>
          <mesh position={[0, 7.5, 0]} material={rocketMaterial}>
            <cylinderGeometry args={[0.6, 0.6, 3, 24]} />
          </mesh>
          
          <mesh position={[0, 6.1, 0]} rotation={[Math.PI / 2, 0, 0]} geometry={ringGeo} material={darkMetal} />

          <mesh position={[0, 5.8, 0]} material={darkMetal}>
            <cylinderGeometry args={[0.2, 0.5, 0.4, 24]} />
          </mesh>
          
          <mesh ref={exhaust2Ref} position={[0, 4.8, 0]} visible={false} material={vacFlame}>
            <coneGeometry args={[0.8, 2, 24]} />
            <pointLight ref={light2Ref} color="#00b4d8" distance={25} decay={2} intensity={0} />
          </mesh>
        </group>
      </group>

      {/* STAGE 1 (Core Booster) */}
      <group ref={stage1WrapperRef}>
        <group ref={stage1GroupRef}>
          <mesh position={[0, 3, 0]} material={rocketMaterial}>
            <cylinderGeometry args={[0.6, 0.6, 6, 32]} />
          </mesh>
          
          <mesh position={[0, 5.9, 0]} rotation={[Math.PI / 2, 0, 0]} geometry={ringGeo} material={darkMetal} />
          <mesh position={[0, 3, 0]} rotation={[Math.PI / 2, 0, 0]} geometry={ringGeo} material={darkMetal} />
          <mesh position={[0, 0.2, 0]} rotation={[Math.PI / 2, 0, 0]} geometry={ringGeo} material={darkMetal} />

          <mesh position={[0, -0.2, 0]} material={darkMetal}>
            <cylinderGeometry args={[0.4, 0.7, 0.6, 24]} />
          </mesh>

          <mesh ref={exhaust1Ref} position={[0, -1.8, 0]} visible={false} material={engineFlame}>
            <coneGeometry args={[0.6, 3, 24]} />
            <pointLight ref={light1Ref} color="#ff7700" distance={40} decay={1.8} intensity={0} />
          </mesh>

          {/* Booster Aerodynamic Fins */}
          <group position={[0, 0.5, 0]}>
            {[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map((rot, i) => (
              <mesh key={i} rotation={[0, rot, 0]} position={[Math.sin(rot) * 0.7, 0, Math.cos(rot) * 0.7]} material={rocketMaterial}>
                <boxGeometry args={[0.08, 1.2, 0.7]} />
              </mesh>
            ))}
          </group>
        </group>
      </group>
    </group>
  );
}

export default function RocketLaunchSequence() {
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);

  const jumpToStage = (offset: number) => {
    window.dispatchEvent(new CustomEvent('jump-to-stage', { detail: { offset } }));
  };

  return (
    <div className="w-full h-full min-h-[100dvh] bg-black relative overflow-hidden select-none font-mono">
      {/* 3D WEBGL CANVAS */}
      <Canvas
        gl={{
          antialias: false,
          powerPreference: "high-performance",
          alpha: false,
          stencil: false,
          depth: true
        }}
        dpr={[1, 1.5]}
        performance={{ min: 0.5 }}
        camera={{ position: [0, 6, 25], fov: 45, near: 0.1, far: 500 }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[50, 20, 20]} intensity={3.0} color="#fffcf5" />
          <directionalLight position={[-30, -20, -10]} intensity={0.9} color="#0088ff" />
          <pointLight position={[0, 0, 15]} intensity={0.6} color="#ffffff" />
          
          <Stars radius={100} depth={50} count={3500} factor={4} saturation={0} fade speed={1.2} />
          
          <ScrollControls pages={5} damping={0.12}>
            <LaunchController isAutoPlaying={isAutoPlaying} setIsAutoPlaying={setIsAutoPlaying} />
          </ScrollControls>

          <EffectComposer disableNormalPass multisampling={0}>
            <Bloom luminanceThreshold={0.55} mipmapBlur intensity={1.2} />
            <Noise opacity={0.015} />
            <Vignette eskil={false} offset={0.1} darkness={1.1} />
          </EffectComposer>
        </Suspense>
      </Canvas>

      {/* TOP TELEMETRY HUD BAR (Direct 60fps DOM Sync) */}
      <div className="absolute top-16 left-0 w-full px-4 sm:px-8 py-2 z-30 pointer-events-none flex flex-wrap items-center justify-between gap-3 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center gap-4 text-xs">
          <div className="flex flex-col">
            <span className="text-[10px] text-white/40 tracking-wider">MISSION TIME</span>
            <span id="hud-met" className="text-white font-bold tracking-widest text-sm">T+ 00:00:00</span>
          </div>
          <div className="w-[1px] h-6 bg-white/20" />
          <div className="flex flex-col">
            <span className="text-[10px] text-white/40 tracking-wider">ALTITUDE</span>
            <span id="hud-altitude" className="text-white font-bold tracking-widest text-sm">0.0 KM</span>
          </div>
          <div className="w-[1px] h-6 bg-white/20" />
          <div className="flex flex-col">
            <span className="text-[10px] text-white/40 tracking-wider">VELOCITY</span>
            <span id="hud-velocity" className="text-white font-bold tracking-widest text-sm">0.00 KM/S</span>
          </div>
        </div>

        {/* Current Flight Status Badge */}
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-white/70 animate-ping" />
          <span id="hud-status" className="text-[11px] sm:text-xs text-white/90 font-bold tracking-widest bg-white/5 border border-white/10 px-3 py-1 rounded-full uppercase backdrop-blur-md">
            PAD T-0 // BOOSTER IGNITION &amp; LIFTOFF
          </span>
        </div>
      </div>

      {/* INTERACTIVE MILESTONES & CONTROLS FOOTER */}
      <div className="absolute bottom-4 sm:bottom-6 left-0 w-full px-4 sm:px-8 z-30 pointer-events-auto flex flex-col gap-2.5">
        {/* Stage Timeline Navigation Pills */}
        <div className="w-full flex items-center justify-between gap-2 overflow-x-auto p-1.5 bg-black/60 backdrop-blur-xl border border-white/10 rounded-xl">
          <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
            {STAGE_MILESTONES.map((stage, i) => (
              <button
                key={stage.id}
                id={`milestone-pill-${i}`}
                type="button"
                onClick={() => jumpToStage(stage.offset)}
                className="px-2.5 py-1 rounded font-mono text-[10px] tracking-wider uppercase bg-black/40 text-white/50 hover:text-white hover:bg-white/10 border border-white/5 transition-all cursor-pointer whitespace-nowrap"
              >
                {stage.label}
              </button>
            ))}
          </div>

          {/* Auto Play / Pause Toggle Button */}
          <div className="flex items-center gap-2 shrink-0 ml-auto">
            <button
              type="button"
              onClick={() => setIsAutoPlaying(!isAutoPlaying)}
              className={`px-3 py-1 rounded font-mono text-[10px] tracking-wider uppercase flex items-center gap-1.5 transition-all cursor-pointer font-bold ${
                isAutoPlaying
                  ? 'bg-white text-black shadow-lg shadow-white/20'
                  : 'bg-white/10 text-white hover:bg-white/20'
              }`}
            >
              <span>{isAutoPlaying ? '⏸ PAUSE' : '▶ AUTO PLAY'}</span>
            </button>
          </div>
        </div>

        {/* Global Progress Bar */}
        <div className="w-full flex items-center gap-3">
          <div className="relative flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div id="hud-progress-bar" className="h-full bg-white rounded-full w-0 transition-all duration-75" />
          </div>
          <span id="hud-percent" className="text-[10px] text-white/50 tracking-wider shrink-0 w-10 text-right">0.0%</span>
        </div>

        {/* Hint */}
        <div className="text-center text-[10px] text-white/40 tracking-[0.2em] uppercase pointer-events-none">
          USE MOUSEWHEEL OR TOUCH TO SCRUB TRAJECTORY • CLICK STAGES TO JUMP
        </div>
      </div>
    </div>
  );
}
