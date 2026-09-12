"use client";

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ScrollControls, useScroll, Stars, useTexture, Environment } from '@react-three/drei';
import { EffectComposer, Bloom, ToneMapping } from '@react-three/postprocessing';
import * as THREE from 'three';

// Utility to create a timeline progress between two offsets [start, end]
function getProgress(offset: number, start: number, end: number) {
  return Math.max(0, Math.min(1, (offset - start) / (end - start)));
}

function LaunchAnimation() {
  const scroll = useScroll();
  
  // Scenery Refs
  const earthGroupRef = useRef<THREE.Group>(null);
  const cloudsRef = useRef<THREE.Mesh>(null);
  
  // Wrappers (for world-space trajectory)
  const stage1WrapperRef = useRef<THREE.Group>(null);
  const stage2WrapperRef = useRef<THREE.Group>(null);
  const fairingLeftWrapperRef = useRef<THREE.Group>(null);
  const fairingRightWrapperRef = useRef<THREE.Group>(null);
  const satelliteWrapperRef = useRef<THREE.Group>(null);
  
  // Local Groups (for local animations like falling, separating, spinning)
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

  const earthTexture = useTexture('/textures/2k_earth_daymap.jpg');
  const cloudsTexture = useTexture('/textures/2k_earth_clouds.jpg');

  useFrame((state, delta) => {
    const offset = scroll.offset;
    
    // Rotate Earth slowly on its axis
    if (earthGroupRef.current) earthGroupRef.current.rotation.y += 0.02 * delta;
    if (cloudsRef.current) cloudsRef.current.rotation.y += 0.03 * delta;

    // --- TIMINGS --- //
    // 0.00 - 0.20: Liftoff
    // 0.20 - 0.30: Max-Q, Pitch over begins
    // 0.30: Stage 1 burnout & sep
    // 0.35: Stage 2 ignition
    // 0.40: Fairing sep
    // 0.50 - 0.70: Orbit insertion curve completes
    // 0.70: Stage 2 cutoff & sep
    // 0.80 - 0.90: Satellite acquires / panels deploy
    // 0.90 - 1.00: Orbiting

    const pMaxQ = getProgress(offset, 0.2, 0.3);         

    // Shared trajectory helper
    const getTrajectory = (clampedOffset: number, pOrbitContinuous: number, time: number) => {
      const pLiftoff = getProgress(clampedOffset, 0.0, 0.25);
      const pPitchOver = getProgress(clampedOffset, 0.2, 0.5);
      const pOrbitInsert = getProgress(clampedOffset, 0.4, 0.7);
      
      let orbitAngle = 0;
      let posX = 0;
      // Start altitude is Y=-1, climbs to Y=19
      let posY = -1 + (pLiftoff * 20);
      let rotZ = -pPitchOver * (Math.PI / 2);

      if (pOrbitInsert > 0) {
        // Curve around earth
        orbitAngle = pOrbitInsert * (Math.PI / 4) + (pOrbitContinuous * time * 0.2);
        const radius = 39; 
        posX = Math.sin(orbitAngle) * radius;
        posY = -20 + Math.cos(orbitAngle) * radius;
        rotZ = -orbitAngle;
      }

      // Max-Q vibration applies to all attached components evenly
      if (pMaxQ > 0 && pMaxQ < 1) {
        posX += (Math.random() - 0.5) * 0.1;
      }

      return { posX, posY, rotZ };
    };

    // --- SATELLITE (Main Payload) ---
    const pOrbit = getProgress(offset, 0.9, 1.0);
    const satTraj = getTrajectory(offset, pOrbit, state.clock.elapsedTime);
    
    if (satelliteWrapperRef.current) {
      satelliteWrapperRef.current.position.set(satTraj.posX, satTraj.posY, 0);
      satelliteWrapperRef.current.rotation.z = satTraj.rotZ;
    }

    const pSatSep = getProgress(offset, 0.7, 0.75); // separation impulse
    const pSatAcq = getProgress(offset, 0.8, 0.9);
    
    if (satelliteGroupRef.current) {
      // Sat separation: it pushes forward slightly from Stage 2.
      // Since local Y points forward, positive Y is the correct axis.
      satelliteGroupRef.current.position.y = (pSatSep * 3); 
      
      if (pSatSep > 0) {
        satelliteGroupRef.current.rotation.y += 0.5 * delta; // stabilize spin
        satelliteGroupRef.current.rotation.x = THREE.MathUtils.lerp(0, Math.PI / 4, pSatAcq);
      }
    }

    // --- STAGE 1 (Booster) ---
    // Freeze at 0.30
    const s1Offset = Math.min(offset, 0.30);
    const s1Traj = getTrajectory(s1Offset, 0, 0);
    if (stage1WrapperRef.current) {
      stage1WrapperRef.current.position.set(s1Traj.posX, s1Traj.posY, 0);
      stage1WrapperRef.current.rotation.z = s1Traj.rotZ;
    }

    const pStage1Sep = getProgress(offset, 0.30, 0.60); 
    if (stage1GroupRef.current) {
      // Local -Y moves it backwards away from the rocket, perfectly mirroring gravity and momentum
      stage1GroupRef.current.position.y = -pStage1Sep * 40; 
      stage1GroupRef.current.position.x = -pStage1Sep * 2; // slight drift
      stage1GroupRef.current.rotation.z = pStage1Sep * 2; // tumble
    }

    // --- FAIRINGS ---
    // Freeze at 0.40
    const fairingOffset = Math.min(offset, 0.40);
    const fairingTraj = getTrajectory(fairingOffset, 0, 0);
    if (fairingLeftWrapperRef.current && fairingRightWrapperRef.current) {
      fairingLeftWrapperRef.current.position.set(fairingTraj.posX, fairingTraj.posY, 0);
      fairingLeftWrapperRef.current.rotation.z = fairingTraj.rotZ;
      fairingRightWrapperRef.current.position.set(fairingTraj.posX, fairingTraj.posY, 0);
      fairingRightWrapperRef.current.rotation.z = fairingTraj.rotZ;
    }

    const pFairingSep = getProgress(offset, 0.40, 0.70);
    if (fairingLeftRef.current && fairingRightRef.current) {
      fairingLeftRef.current.position.x = -pFairingSep * 10;
      fairingLeftRef.current.position.y = -pFairingSep * 30; // drop behind
      fairingLeftRef.current.rotation.z = pFairingSep * 3;

      fairingRightRef.current.position.x = pFairingSep * 10;
      fairingRightRef.current.position.y = -pFairingSep * 30;
      fairingRightRef.current.rotation.z = -pFairingSep * 3;
    }

    // --- STAGE 2 (Upper Stage) ---
    // Freeze at 0.70
    const s2Offset = Math.min(offset, 0.70);
    const s2Traj = getTrajectory(s2Offset, 0, 0);
    if (stage2WrapperRef.current) {
      stage2WrapperRef.current.position.set(s2Traj.posX, s2Traj.posY, 0);
      stage2WrapperRef.current.rotation.z = s2Traj.rotZ;
    }

    const pStage2Sep = getProgress(offset, 0.70, 1.0);
    if (stage2GroupRef.current) {
      stage2GroupRef.current.position.y = -pStage2Sep * 30; 
      stage2GroupRef.current.rotation.z = pStage2Sep * 0.5; // slight tumble
    }

    // CAMERA LOGIC
    let targetCamPos = new THREE.Vector3();
    let targetLookAt = new THREE.Vector3();

    if (satelliteWrapperRef.current && satelliteGroupRef.current) {
      const vPos = new THREE.Vector3();
      satelliteGroupRef.current.getWorldPosition(vPos); // Always look at the satellite
      
      if (offset < 0.9) {
        // Follow closely during launch and insertion
        targetCamPos.set(vPos.x, vPos.y + 2, vPos.z + 20);
        targetLookAt.set(vPos.x, vPos.y + 4, vPos.z);
      } else {
        // Grand finale pull-back
        targetCamPos.set(0, -5, 65);
        targetLookAt.set(0, -10, 0);
      }
    }

    state.camera.position.lerp(targetCamPos, 0.04);
    state.camera.lookAt(targetLookAt);

    // EXHAUST 1 (Booster)
    const pStage1Burnout = getProgress(offset, 0.28, 0.30); 
    if (exhaust1Ref.current && light1Ref.current) {
      if (offset > 0 && pStage1Burnout < 1) {
        exhaust1Ref.current.visible = true;
        const intensity = pMaxQ > 0 && pMaxQ < 1 ? 1.5 : (1 - pStage1Burnout);
        exhaust1Ref.current.scale.set(1 + Math.random() * 0.2, (1 + Math.random() * 0.8) * intensity, 1 + Math.random() * 0.2);
        light1Ref.current.intensity = (15 + Math.random() * 10) * intensity;
        (exhaust1Ref.current.material as THREE.MeshStandardMaterial).emissiveIntensity = (5 + Math.random() * 5) * intensity;
      } else {
        exhaust1Ref.current.visible = false;
        light1Ref.current.intensity = 0;
      }
    }

    // EXHAUST 2 (Upper Stage)
    const pStage2Ignition = getProgress(offset, 0.35, 0.40); 
    const pStage2Cutoff = getProgress(offset, 0.65, 0.70); 
    if (exhaust2Ref.current && light2Ref.current) {
      if (pStage2Ignition > 0 && pStage2Cutoff < 1) {
        exhaust2Ref.current.visible = true;
        // Ignites slowly, cuts off quickly
        const intensity = pStage2Ignition * (1 - pStage2Cutoff);
        exhaust2Ref.current.scale.set((1.5 + Math.random() * 0.2) * intensity, (0.8 + Math.random() * 0.4) * intensity, (1.5 + Math.random() * 0.2) * intensity);
        light2Ref.current.intensity = (5 + Math.random() * 5) * intensity;
        (exhaust2Ref.current.material as THREE.MeshStandardMaterial).emissiveIntensity = (3 + Math.random() * 2) * intensity;
      } else {
        exhaust2Ref.current.visible = false;
        light2Ref.current.intensity = 0;
      }
    }

    // SOLAR PANELS
    if (leftPanelRef.current && rightPanelRef.current) {
      leftPanelRef.current.position.x = THREE.MathUtils.lerp(0, -1.8, pSatAcq);
      leftPanelRef.current.rotation.y = THREE.MathUtils.lerp(Math.PI/2, 0, pSatAcq);
      
      rightPanelRef.current.position.x = THREE.MathUtils.lerp(0, 1.8, pSatAcq);
      rightPanelRef.current.rotation.y = THREE.MathUtils.lerp(-Math.PI/2, 0, pSatAcq);
    }
  });

  // PBR Materials
  const rocketMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#ffffff', metalness: 0.7, roughness: 0.2, envMapIntensity: 2.0 }), []);
  const darkMetal = useMemo(() => new THREE.MeshStandardMaterial({ color: '#222222', metalness: 0.9, roughness: 0.3, envMapIntensity: 1.5 }), []);
  const goldFoil = useMemo(() => new THREE.MeshStandardMaterial({ color: '#ffcc00', metalness: 0.8, roughness: 0.4, envMapIntensity: 1.5 }), []);
  const engineFlame = useMemo(() => new THREE.MeshStandardMaterial({ color: '#ff5500', emissive: '#ffaa00', emissiveIntensity: 5, transparent: true, opacity: 0.9 }), []);
  const vacFlame = useMemo(() => new THREE.MeshStandardMaterial({ color: '#3388ff', emissive: '#00ccff', emissiveIntensity: 3, transparent: true, opacity: 0.8 }), []);

  // EXACT GEOMETRY ALIGNMENTS
  // Base of the rocket is at Y = 0.
  // Stage 1 (Booster): Y = 0 to 6
  // Stage 2 (Upper): Y = 6 to 9
  // Satellite (Payload): Y = 9 to 11
  // Fairings (Shell): Y = 9 to 12.5

  return (
    <group>
      {/* EARTH SYSTEM */}
      <group ref={earthGroupRef} position={[0, -20, 0]}>
        <mesh>
          <sphereGeometry args={[18, 64, 64]} />
          <meshStandardMaterial map={earthTexture} roughness={0.7} metalness={0.1} />
        </mesh>
        <mesh ref={cloudsRef}>
          <sphereGeometry args={[18.1, 64, 64]} />
          <meshStandardMaterial map={cloudsTexture} transparent opacity={0.6} depthWrite={false} blending={THREE.AdditiveBlending} />
        </mesh>
        <mesh>
          <sphereGeometry args={[18.4, 64, 64]} />
          <meshBasicMaterial color="#4b70dd" transparent opacity={0.15} side={THREE.BackSide} />
        </mesh>
      </group>

      {/* SATELLITE WRAPPER */}
      <group ref={satelliteWrapperRef}>
        <group ref={satelliteGroupRef}>
          <group position={[0, 10, 0]} scale={[0.5, 0.5, 0.5]}>
            <mesh>
              <boxGeometry args={[1.2, 2.5, 1.2]} />
              <primitive object={goldFoil} attach="material" />
            </mesh>
            
            <mesh ref={leftPanelRef} position={[0, 0, 0]}>
              <boxGeometry args={[2, 4, 0.05]} />
              <meshStandardMaterial color="#0a1b2c" metalness={1} roughness={0.2} envMapIntensity={2} />
              <gridHelper args={[2, 10, 0x4488ff, 0x4488ff]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.03]} />
            </mesh>

            <mesh ref={rightPanelRef} position={[0, 0, 0]}>
              <boxGeometry args={[2, 4, 0.05]} />
              <meshStandardMaterial color="#0a1b2c" metalness={1} roughness={0.2} envMapIntensity={2} />
              <gridHelper args={[2, 10, 0x4488ff, 0x4488ff]} rotation={[Math.PI/2, 0, 0]} position={[0, 0, 0.03]} />
            </mesh>

            <mesh position={[0, 1.5, 0]} rotation={[-0.2, 0, 0]}>
              <cylinderGeometry args={[0.8, 0.1, 0.4, 32]} />
              <meshStandardMaterial color="#ffffff" metalness={0.5} roughness={0.5} />
            </mesh>
            
            <mesh position={[0, 1.9, 0]}>
              <cylinderGeometry args={[0.05, 0.05, 1, 16]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>
          </group>
        </group>
      </group>

      {/* FAIRING LEFT WRAPPER */}
      <group ref={fairingLeftWrapperRef}>
        <group ref={fairingLeftRef}>
          <mesh position={[-0.3, 10, 0]}> {/* Base of fairing cylinder */}
            <cylinderGeometry args={[0.6, 0.6, 2, 32, 1, false, Math.PI, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
          <mesh position={[-0.3, 11.75, 0]}> {/* Cone on top */}
            <coneGeometry args={[0.6, 1.5, 32, 1, false, Math.PI, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
        </group>
      </group>

      {/* FAIRING RIGHT WRAPPER */}
      <group ref={fairingRightWrapperRef}>
        <group ref={fairingRightRef}>
          <mesh position={[0.3, 10, 0]}>
            <cylinderGeometry args={[0.6, 0.6, 2, 32, 1, false, 0, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
          <mesh position={[0.3, 11.75, 0]}>
            <coneGeometry args={[0.6, 1.5, 32, 1, false, 0, Math.PI]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
        </group>
      </group>

      {/* STAGE 2 WRAPPER */}
      <group ref={stage2WrapperRef}>
        <group ref={stage2GroupRef}>
          <mesh position={[0, 7.5, 0]}> {/* Height 3, center at 7.5 */}
            <cylinderGeometry args={[0.6, 0.6, 3, 32]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
          
          <mesh position={[0, 6.1, 0]} rotation={[Math.PI/2, 0, 0]}>
            <torusGeometry args={[0.6, 0.02, 16, 64]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>

          <mesh position={[0, 5.8, 0]}> {/* Engine */}
            <cylinderGeometry args={[0.2, 0.5, 0.4, 32]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>
          
          <mesh ref={exhaust2Ref} position={[0, 4.8, 0]} visible={false}>
            <coneGeometry args={[0.8, 2, 32]} />
            <primitive object={vacFlame} attach="material" />
            <pointLight ref={light2Ref} color="#00aaff" distance={30} decay={2} intensity={0} />
          </mesh>
        </group>
      </group>

      {/* STAGE 1 WRAPPER */}
      <group ref={stage1WrapperRef}>
        <group ref={stage1GroupRef}>
          <mesh position={[0, 3, 0]}> {/* Height 6, center at 3 */}
            <cylinderGeometry args={[0.6, 0.6, 6, 64]} />
            <primitive object={rocketMaterial} attach="material" />
          </mesh>
          
          <mesh position={[0, 5.9, 0]} rotation={[Math.PI/2, 0, 0]}>
            <torusGeometry args={[0.6, 0.02, 16, 64]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>
          <mesh position={[0, 3, 0]} rotation={[Math.PI/2, 0, 0]}>
            <torusGeometry args={[0.6, 0.02, 16, 64]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>
          <mesh position={[0, 0.2, 0]} rotation={[Math.PI/2, 0, 0]}>
            <torusGeometry args={[0.6, 0.04, 16, 64]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>

          <mesh position={[0, -0.2, 0]}>
            <cylinderGeometry args={[0.4, 0.7, 0.6, 32]} />
            <primitive object={darkMetal} attach="material" />
          </mesh>

          <mesh ref={exhaust1Ref} position={[0, -1.8, 0]} visible={false}>
            <coneGeometry args={[0.6, 3, 32]} />
            <primitive object={engineFlame} attach="material" />
            <pointLight ref={light1Ref} color="#ff7700" distance={50} decay={1.5} intensity={0} />
          </mesh>

          <group position={[0, 0.5, 0]}>
            {[0, Math.PI/2, Math.PI, Math.PI*1.5].map((rot, i) => (
              <mesh key={i} rotation={[0, rot, 0]} position={[Math.sin(rot)*0.7, 0, Math.cos(rot)*0.7]}>
                <boxGeometry args={[0.1, 1.2, 0.8]} />
                <primitive object={rocketMaterial} attach="material" />
              </mesh>
            ))}
          </group>
        </group>
      </group>
    </group>
  );
}

export default function RocketLaunchSequence() {
  return (
    <div className="w-full h-[100dvh] bg-[#000005] relative flex flex-col">
      <Canvas camera={{ position: [0, 2, 14], fov: 45 }}>
        <Environment preset="night" background={false} />
        <ambientLight intensity={0.2} />
        <directionalLight position={[10, 10, 10]} intensity={2.5} color="#ffeedd" castShadow />
        <directionalLight position={[-10, -10, -10]} intensity={0.5} color="#4b70dd" />
        
        <Stars radius={100} depth={50} count={8000} factor={4} saturation={0} fade speed={1.5} />
        
        {/* Adjusted pages to 6 for snappy but detailed pacing */}
        <ScrollControls pages={6} damping={0.2}>
          <LaunchAnimation />
        </ScrollControls>

        <EffectComposer disableNormalPass>
          <Bloom luminanceThreshold={1} mipmapBlur intensity={1.5} />
          <ToneMapping />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
