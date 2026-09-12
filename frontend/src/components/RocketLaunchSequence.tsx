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
  
  // Vehicle Structure Refs
  const vehicleGroupRef = useRef<THREE.Group>(null);
  const vehicleWrapperRef = useRef<THREE.Group>(null); // For orbiting
  
  // Stage 1
  const stage1GroupRef = useRef<THREE.Group>(null);
  const exhaust1Ref = useRef<THREE.Mesh>(null);
  const light1Ref = useRef<THREE.PointLight>(null);
  
  // Stage 2
  const stage2GroupRef = useRef<THREE.Group>(null);
  const exhaust2Ref = useRef<THREE.Mesh>(null);
  const light2Ref = useRef<THREE.PointLight>(null);
  
  // Fairings
  const fairingLeftRef = useRef<THREE.Group>(null);
  const fairingRightRef = useRef<THREE.Group>(null);
  
  // Satellite
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

    // --- TIMELINES --- //
    const pLiftoff = getProgress(offset, 0.05, 0.3);       
    const pPitchOver = getProgress(offset, 0.15, 0.4);    
    const pMaxQ = getProgress(offset, 0.2, 0.25);         
    const pStage1Burnout = getProgress(offset, 0.25, 0.28); 
    const pStage1Sep = getProgress(offset, 0.3, 0.35);    
    const pStage2Ignition = getProgress(offset, 0.35, 0.65); 
    const pFairingSep = getProgress(offset, 0.4, 0.45);   
    const pOrbitInsert = getProgress(offset, 0.45, 0.7); 
    const pStage2Cutoff = getProgress(offset, 0.65, 0.7); 
    const pSatSep = getProgress(offset, 0.7, 0.8);       
    const pSatAcq = getProgress(offset, 0.8, 0.9);      
    const pOrbit = getProgress(offset, 0.9, 1.0); // Final continuous orbit

    // TRUE TRAJECTORY MATH
    // Earth center is at [0, -20, 0]. Radius is 18. Surface is at Y = -2.
    // Rocket starts at Y = -1.
    
    let currentAltitude = -1 + (pLiftoff * 20); // Climbs to Y=19
    let orbitAngle = 0;

    if (vehicleWrapperRef.current && vehicleGroupRef.current) {
      
      // Pitch over rotates the vehicle to point horizontally
      vehicleGroupRef.current.rotation.z = -pPitchOver * (Math.PI / 2);
      
      if (pOrbitInsert > 0) {
        // Curve around the earth using Sine/Cosine
        // We transition from moving straight up, to orbiting around the center [0, -20, 0]
        orbitAngle = pOrbitInsert * (Math.PI / 4) + (pOrbit * state.clock.elapsedTime * 0.2);
        
        // Radius from earth center. Surface is 18. 
        // Peak altitude is Y=19, meaning distance from center is 39.
        const radius = 39; 
        
        // Calculate position relative to earth center
        const posX = Math.sin(orbitAngle) * radius;
        const posY = -20 + Math.cos(orbitAngle) * radius;
        
        vehicleWrapperRef.current.position.set(posX, posY, 0);
        
        // Orient the vehicle so it points along the orbit tangent
        // When angle is 0 (top of earth), rotation should be -PI/2 (horizontal)
        // As angle increases, rotation must decrease
        vehicleWrapperRef.current.rotation.z = -orbitAngle;
      } else {
        // Vertical flight
        vehicleWrapperRef.current.position.set(0, currentAltitude, 0);
        vehicleWrapperRef.current.rotation.z = 0;
      }

      // Vibration during Max-Q
      if (pMaxQ > 0 && pMaxQ < 1) {
        vehicleGroupRef.current.position.x = (Math.random() - 0.5) * 0.1;
      } else {
        vehicleGroupRef.current.position.x = 0;
      }
    }

    // CAMERA LOGIC
    // Follow the vehicle, then pull back to see the whole earth
    let targetCamPos = new THREE.Vector3();
    let targetLookAt = new THREE.Vector3();

    if (vehicleWrapperRef.current) {
      const vPos = vehicleWrapperRef.current.position;
      
      if (pSatSep < 1) {
        // Follow closely during launch and insertion
        targetCamPos.set(vPos.x, vPos.y + 2, vPos.z + 20);
        targetLookAt.set(vPos.x, vPos.y + 2, vPos.z);
      } else {
        // Pull back to reveal Earth and satellite orbiting
        // Camera moves to a grand vantage point
        targetCamPos.set(0, -5, 65);
        targetLookAt.set(0, -10, 0);
      }
    }

    state.camera.position.lerp(targetCamPos, 0.03);
    state.camera.lookAt(targetLookAt);

    // STAGE 1 LOGIC
    if (stage1GroupRef.current) {
      if (pStage1Sep > 0) {
        // Stage 1 falls back towards Earth (Y decreases relative to wrapper)
        stage1GroupRef.current.position.y = -pStage1Sep * 25;
        stage1GroupRef.current.position.x = -pStage1Sep * 5;
        stage1GroupRef.current.rotation.z = pStage1Sep * 1.5;
      }
    }

    // EXHAUST 1 (Booster)
    if (exhaust1Ref.current && light1Ref.current) {
      if (pLiftoff > 0 && pStage1Burnout < 1) {
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
    if (exhaust2Ref.current && light2Ref.current) {
      if (pStage2Ignition > 0 && pStage2Cutoff < 1) {
        exhaust2Ref.current.visible = true;
        exhaust2Ref.current.scale.set(1.5 + Math.random() * 0.2, 0.8 + Math.random() * 0.4, 1.5 + Math.random() * 0.2);
        light2Ref.current.intensity = 5 + Math.random() * 5;
        (exhaust2Ref.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 3 + Math.random() * 2;
      } else {
        exhaust2Ref.current.visible = false;
        light2Ref.current.intensity = 0;
      }
    }

    // FAIRING LOGIC
    if (fairingLeftRef.current && fairingRightRef.current) {
      fairingLeftRef.current.position.x = -pFairingSep * 6;
      fairingLeftRef.current.position.y = -pFairingSep * 4;
      fairingLeftRef.current.rotation.z = pFairingSep * 1;
      
      fairingRightRef.current.position.x = pFairingSep * 6;
      fairingRightRef.current.position.y = -pFairingSep * 4;
      fairingRightRef.current.rotation.z = -pFairingSep * 1;
    }

    // SATELLITE SEPARATION
    if (satelliteGroupRef.current) {
      // Sat pushes forward from the second stage
      satelliteGroupRef.current.position.y = 1.5 + (pSatSep * 5);
      
      if (pSatSep > 0) {
        // Spin stabilizes
        satelliteGroupRef.current.rotation.y += 0.5 * delta;
        satelliteGroupRef.current.rotation.x = THREE.MathUtils.lerp(0, Math.PI / 4, pSatAcq);
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

      {/* VEHICLE WRAPPER (Handles overall position in world space) */}
      <group ref={vehicleWrapperRef}>
        
        {/* VEHICLE GROUP (Handles pitch over) */}
        <group ref={vehicleGroupRef}>
          
          {/* === STAGE 1 (Booster) === */}
          <group ref={stage1GroupRef}>
            <mesh position={[0, -1.5, 0]}>
              <cylinderGeometry args={[0.6, 0.6, 6, 64]} />
              <primitive object={rocketMaterial} attach="material" />
            </mesh>
            
            <mesh position={[0, 1.4, 0]} rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[0.6, 0.02, 16, 64]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>
            <mesh position={[0, -1.5, 0]} rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[0.6, 0.02, 16, 64]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>
            <mesh position={[0, -4.4, 0]} rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[0.6, 0.04, 16, 64]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>

            <mesh position={[0, -4.5, 0]}>
              <cylinderGeometry args={[0.4, 0.7, 0.6, 32]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>
            <mesh ref={exhaust1Ref} position={[0, -5.8, 0]} visible={false}>
              <coneGeometry args={[0.6, 3, 32]} />
              <primitive object={engineFlame} attach="material" />
              <pointLight ref={light1Ref} color="#ff7700" distance={50} decay={1.5} intensity={0} />
            </mesh>
            <group position={[0, -3.8, 0]}>
              {[0, Math.PI/2, Math.PI, Math.PI*1.5].map((rot, i) => (
                <mesh key={i} rotation={[0, rot, 0]} position={[Math.sin(rot)*0.7, 0, Math.cos(rot)*0.7]}>
                  <boxGeometry args={[0.1, 1.2, 0.8]} />
                  <primitive object={rocketMaterial} attach="material" />
                </mesh>
              ))}
            </group>
          </group>

          {/* === STAGE 2 (Upper Stage & Payload) === */}
          <group ref={stage2GroupRef} position={[0, 1.5, 0]}>
            <mesh position={[0, -0.5, 0]}>
              <cylinderGeometry args={[0.6, 0.6, 2, 32]} />
              <primitive object={rocketMaterial} attach="material" />
            </mesh>
            
            <mesh position={[0, -1.4, 0]} rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[0.6, 0.02, 16, 64]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>

            <mesh position={[0, -1.6, 0]}>
              <cylinderGeometry args={[0.2, 0.5, 0.4, 32]} />
              <primitive object={darkMetal} attach="material" />
            </mesh>
            <mesh ref={exhaust2Ref} position={[0, -2.5, 0]} visible={false}>
              <coneGeometry args={[0.8, 2, 32]} />
              <primitive object={vacFlame} attach="material" />
              <pointLight ref={light2Ref} color="#00aaff" distance={30} decay={2} intensity={0} />
            </mesh>

            {/* FAIRING LEFT */}
            <group ref={fairingLeftRef} position={[0, 0.5, 0]}>
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
            <group ref={fairingRightRef} position={[0, 0.5, 0]}>
              <mesh position={[0.3, 0.5, 0]}>
                <cylinderGeometry args={[0.6, 0.6, 2, 32, 1, false, 0, Math.PI]} />
                <primitive object={rocketMaterial} attach="material" />
              </mesh>
              <mesh position={[0.3, 2, 0]}>
                <coneGeometry args={[0.6, 1.5, 32, 1, false, 0, Math.PI]} />
                <primitive object={rocketMaterial} attach="material" />
              </mesh>
            </group>

            {/* SATELLITE PAYLOAD */}
            <group ref={satelliteGroupRef} position={[0, 1.5, 0]} scale={[0.5, 0.5, 0.5]}>
              <mesh>
                <boxGeometry args={[1.2, 1.5, 1.2]} />
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

              <mesh position={[0, 1.0, 0]} rotation={[-0.2, 0, 0]}>
                <cylinderGeometry args={[0.8, 0.1, 0.4, 32]} />
                <meshStandardMaterial color="#ffffff" metalness={0.5} roughness={0.5} />
              </mesh>
              
              <mesh position={[0, 1.4, 0]}>
                <cylinderGeometry args={[0.05, 0.05, 1, 16]} />
                <primitive object={darkMetal} attach="material" />
              </mesh>
            </group>
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
