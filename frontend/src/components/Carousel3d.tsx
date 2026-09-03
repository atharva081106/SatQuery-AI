"use client";

import React, { startTransition, useEffect, useRef, useState } from "react";
import * as THREE from "three";

const MAX_SLIDES = 10;
const SCROLL_SMOOTHING = 0.05;
const MOMENTUM_FRICTION = 0.952;
const WHEEL_SPEED = 0.01;
const DRAG_SPEED = 0.01;
const MAX_DPR = 1.5;
const BASE_SLIDE_HEIGHT = 1.5;
const DEFAULT_SLIDE_SCALE = 1.2;

const wrap = (value: number, range: number) => ((value % range) + range) % range;

const parseColor = (color: string) => {
  const rgba = color.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/);
  if (rgba) return `rgb(${rgba[1]}, ${rgba[2]}, ${rgba[3]})`;
  const hsla = color.match(/hsla?\(\s*([\d.]+)\s*,\s*([\d.%]+)\s*,\s*([\d.%]+)/);
  if (hsla) return `hsl(${hsla[1]}, ${hsla[2]}, ${hsla[3]})`;
  return color;
};

export default function Carousel3d({
  slides = [],
  backgroundColor = "transparent",
  direction = "horizontal",
  borderRadius = 0.05,
  slideAspectRatio = 1.5,
  slideScale = DEFAULT_SLIDE_SCALE,
  gap = 0.1,
  activeScale = 1.1,
  effectPreset = "cylinder",
  effectPerspective = 45,
  effectRotation = 40,
  effectDepth = 2.5,
  autoplayEnabled = true,
  autoplaySpeed = 20,
  interactive = true,
  style = {}
}: any) {
  const outerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [activeSlide, setActiveSlide] = useState(0);
  const [webglFailed, setWebglFailed] = useState(false);
  const activeSlideIndexRef = useRef(0);
  const prefersReducedMotionRef = useRef(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => {
      prefersReducedMotionRef.current = mq.matches;
    };
    update();
    mq.addEventListener?.("change", update);
    return () => mq.removeEventListener?.("change", update);
  }, []);

  const configRef = useRef({
    backgroundColor, direction, slideAspectRatio, slideHeightUnit: BASE_SLIDE_HEIGHT * slideScale,
    slideGap: gap, effectPreset, effectPerspective, effectRotation, effectDepth,
    borderRadius, activeScale, autoplayEnabled, autoplaySpeed, interactive
  });
  
  configRef.current = {
    backgroundColor, direction, slideAspectRatio, slideHeightUnit: BASE_SLIDE_HEIGHT * slideScale,
    slideGap: gap, effectPreset, effectPerspective, effectRotation, effectDepth,
    borderRadius, activeScale, autoplayEnabled, autoplaySpeed, interactive
  };

  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;
    
    const validSlides = slides.filter((s: any) => s.src);
    if (validSlides.length === 0) return;
    
    const totalSlides = validSlides.length;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    } catch {
      startTransition(() => setWebglFailed(true));
      return;
    }
    
    const getSize = () => {
      const rect = container.getBoundingClientRect();
      return { width: Math.max(1, Math.floor(rect.width)), height: Math.max(1, Math.floor(rect.height)) };
    };
    
    const size = getSize();
    renderer.setSize(size.width, size.height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR));
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    container.appendChild(renderer.domElement);
    
    let disposed = false;
    const scene = new THREE.Scene();
    
    if (backgroundColor && backgroundColor !== "transparent") {
      scene.background = new THREE.Color(parseColor(backgroundColor));
    } else {
      scene.background = null;
    }
    
    const camera = new THREE.PerspectiveCamera(configRef.current.effectPerspective, size.width / size.height, 0.1, 100);
    camera.position.z = 4.2;
    
    const slideHeights = Array.from({ length: totalSlides }, () => configRef.current.slideHeightUnit);
    const slideSizes = slideHeights.map(h => (configRef.current.direction === "horizontal" ? h * configRef.current.slideAspectRatio : h));
    const slideOffsets: number[] = [];
    let stackPosition = 0;
    
    for (let i = 0; i < totalSlides; i++) {
      if (i === 0) {
        slideOffsets.push(0);
        stackPosition = slideSizes[0] / 2;
      } else {
        stackPosition += configRef.current.slideGap + slideSizes[i] / 2;
        slideOffsets.push(stackPosition);
        stackPosition += slideSizes[i] / 2;
      }
    }
    
    const loopLength = stackPosition + configRef.current.slideGap + slideSizes[0] / 2;
    const halfLoop = loopLength / 2;

    const coverVertexShader = `
      varying vec2 vUv;
      void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;
    const coverFragmentShader = `
      uniform sampler2D uTexture;
      uniform vec2 uPlaneAspect;
      uniform vec2 uImageAspect;
      uniform float uOpacity;
      uniform float uBorderRadius;
      varying vec2 vUv;
      void main() {
          vec2 uv = vUv;
          float planeRatio = uPlaneAspect.x / uPlaneAspect.y;
          float imageRatio = uImageAspect.x / uImageAspect.y;
          if (planeRatio > imageRatio) {
              float scale = imageRatio / planeRatio;
              uv.y = uv.y * scale + (1.0 - scale) * 0.5;
          } else {
              float scale = planeRatio / imageRatio;
              uv.x = uv.x * scale + (1.0 - scale) * 0.5;
          }
          vec4 texColor = texture2D(uTexture, uv);
          vec4 placeholder = vec4(0.0, 0.0, 0.0, 0.0);
          gl_FragColor = mix(placeholder, texColor, uOpacity);

          if (uBorderRadius > 0.0) {
              float ar = uPlaneAspect.x / uPlaneAspect.y;
              vec2 uvCorr = (vUv - 0.5) * vec2(ar, 1.0);
              vec2 halfSize = vec2(ar * 0.5, 0.5);
              float r = uBorderRadius;
              vec2 q = abs(uvCorr) - halfSize + r;
              float d = length(max(q, 0.0)) - r;
              gl_FragColor.a *= 1.0 - smoothstep(-0.003, 0.003, d);
          }
      }
    `;

    const meshes: THREE.Mesh[] = [];
    const textureLoader = new THREE.TextureLoader();
    textureLoader.crossOrigin = "anonymous";
    
    for (let i = 0; i < totalSlides; i++) {
      const height = slideHeights[i];
      const width = height * configRef.current.slideAspectRatio;
      const geometry = new THREE.PlaneGeometry(width, height, 32, 16);
      const material = new THREE.ShaderMaterial({
        vertexShader: coverVertexShader,
        fragmentShader: coverFragmentShader,
        uniforms: {
          uTexture: { value: null },
          uPlaneAspect: { value: new THREE.Vector2(width, height) },
          uImageAspect: { value: new THREE.Vector2(1, 1) },
          uOpacity: { value: 0 },
          uBorderRadius: { value: borderRadius }
        },
        transparent: true,
        side: THREE.DoubleSide
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData = { originalVertices: [...geometry.attributes.position.array], offset: slideOffsets[i], index: i };
      
      const imgUrl = validSlides[i].src;
      if (imgUrl) {
        textureLoader.load(imgUrl, texture => {
          if (disposed) { texture.dispose(); return; }
          material.uniforms.uTexture.value = texture;
          material.uniforms.uImageAspect.value.set(texture.image.width, texture.image.height);
          mesh.userData.opacityTarget = 1;
        });
      }
      scene.add(mesh);
      meshes.push(mesh);
    }

    function applyCoverflowEffect(mesh: THREE.Mesh, pos: number, cfg: any, horizontal: boolean) {
      const maxRot = (cfg.effectRotation * Math.PI) / 180;
      const t = Math.max(-1, Math.min(1, pos / 1.5));
      const absT = Math.abs(t);
      if (horizontal) {
        mesh.rotation.y = -t * maxRot;
        mesh.position.z = -absT * cfg.effectDepth * 0.5;
      } else {
        mesh.rotation.x = t * maxRot;
        mesh.position.z = -absT * cfg.effectDepth * 0.5;
      }
    }

    function applyCylinderEffect(mesh: THREE.Mesh, pos: number, cfg: any, horizontal: boolean) {
      const anglePerUnit = 0.75;
      const angle = pos * anglePerUnit;
      const clamped = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, angle));
      const radius = 1 + cfg.effectDepth;
      mesh.position.z = -radius * Math.cos(clamped);
      const rotScale = (cfg.effectRotation / 45) * 0.9;
      if (horizontal) {
        mesh.rotation.y = -clamped * rotScale;
      } else {
        mesh.rotation.x = clamped * rotScale;
      }
      const positions = mesh.geometry.attributes.position;
      const original = mesh.userData.originalVertices;
      const bow = cfg.effectDepth * 0.12;
      for (let i = 0; i < positions.count; i++) {
        const x = original[i * 3];
        const y = original[i * 3 + 1];
        const axis = horizontal ? x : y;
        const z = -(axis * axis) * bow;
        positions.setZ(i, z);
      }
      positions.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
      mesh.userData.verticesBent = true;
    }

    let scrollPosition = 0, scrollTarget = 0, scrollMomentum = 0;
    let isScrolling = false, lastFrameTime = 0, isDragging = false;
    let dragStartX = 0, dragStartY = 0, dragDelta = 0, totalDragDistance = 0;
    let touchStartX = 0, touchStartY = 0, touchLastX = 0, touchLastY = 0, touchTotalDistance = 0;
    let activeSlideIndex = -1, scrollTimeout: any = null, isHovered = false, autoplayIdleTime = 0;
    
    const raycaster = new THREE.Raycaster();
    const pointerNDC = new THREE.Vector2();

    const getClickedSlide = (clientX: number, clientY: number) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointerNDC.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      pointerNDC.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNDC, camera);
      const hits = raycaster.intersectObjects(meshes);
      if (hits.length > 0) return hits[0].object.userData.index;
      return null;
    };

    const scrollToSlide = (targetIndex: number) => {
      const targetOffset = slideOffsets[targetIndex];
      const wrappedPos = wrap(scrollTarget, loopLength);
      let delta = targetOffset - wrappedPos;
      if (delta > halfLoop) delta -= loopLength;
      if (delta < -halfLoop) delta += loopLength;
      scrollTarget += delta;
      isScrolling = true;
      if (scrollTimeout) clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => isScrolling = false, 600);
    };

    const getCenteredSlideIndex = () => {
      const wrappedPos = wrap(scrollPosition, loopLength);
      let bestIdx = 0, bestDist = Infinity;
      for (let i = 0; i < totalSlides; i++) {
        let d = slideOffsets[i] - wrappedPos;
        if (d > halfLoop) d -= loopLength;
        if (d < -halfLoop) d += loopLength;
        if (Math.abs(d) < bestDist) { bestDist = Math.abs(d); bestIdx = i; }
      }
      return bestIdx;
    };

    const snapOnRelease = (dragMagnitude: number, dragSign: number) => {
      const centered = getCenteredSlideIndex();
      if (dragMagnitude < 25 || dragSign === 0) { scrollToSlide(centered); return; }
      const step = dragSign > 0 ? -1 : 1;
      scrollToSlide(wrap(centered + step, totalSlides));
    };

    const onWheel = (e: WheelEvent) => {
      if (!configRef.current.interactive) return;
      e.preventDefault();
      const rawDelta = configRef.current.direction === "horizontal" 
        ? (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) 
        : e.deltaY;
      scrollTarget += Math.sign(rawDelta) * Math.min(Math.abs(rawDelta), 150) * WHEEL_SPEED;
      isScrolling = true;
      if (scrollTimeout) clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => isScrolling = false, 150);
    };

    const onTouchStart = (e: TouchEvent) => {
      touchStartX = touchLastX = e.touches[0].clientX;
      touchStartY = touchLastY = e.touches[0].clientY;
      touchTotalDistance = 0; scrollMomentum = 0;
      if (scrollTimeout) clearTimeout(scrollTimeout);
    };
    
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const delta = configRef.current.direction === "horizontal" ? e.touches[0].clientX - touchLastX : e.touches[0].clientY - touchLastY;
      touchLastX = e.touches[0].clientX; touchLastY = e.touches[0].clientY;
      touchTotalDistance += Math.abs(delta);
      scrollTarget -= delta * 0.01;
      isScrolling = true;
    };
    
    const onTouchEnd = () => {
      if (touchTotalDistance < 10) {
        const clicked = getClickedSlide(touchStartX, touchStartY);
        if (clicked !== null && clicked !== activeSlideIndexRef.current) scrollToSlide(clicked);
      }
      const swipeDelta = configRef.current.direction === "horizontal" ? touchLastX - touchStartX : touchLastY - touchStartY;
      snapOnRelease(touchTotalDistance, Math.sign(swipeDelta));
    };
    
    const onPointerDown = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      isDragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
      dragDelta = 0; totalDragDistance = 0; scrollMomentum = 0;
      renderer.domElement.style.cursor = "grabbing";
    };
    
    const onPointerMove = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      if (!isDragging) {
        renderer.domElement.style.cursor = "grab"; return;
      }
      const delta = configRef.current.direction === "horizontal" ? e.clientX - dragStartX : e.clientY - dragStartY;
      dragStartX = e.clientX; dragStartY = e.clientY;
      dragDelta = delta; totalDragDistance += Math.abs(delta);
      scrollTarget -= delta * DRAG_SPEED;
      isScrolling = true;
    };
    
    const onPointerUp = (e: PointerEvent) => {
      if (!isDragging || e.pointerType === "touch") return;
      isDragging = false; renderer.domElement.style.cursor = "grab";
      if (totalDragDistance < 10) {
        const clicked = getClickedSlide(e.clientX, e.clientY);
        if (clicked !== null && clicked !== activeSlideIndexRef.current) scrollToSlide(clicked);
        return;
      }
      snapOnRelease(totalDragDistance, Math.sign(dragDelta));
    };

    container.addEventListener("mouseenter", () => { isHovered = true; });
    container.addEventListener("mouseleave", () => { isHovered = false; autoplayIdleTime = 0; });
    container.addEventListener("wheel", onWheel, { passive: false });
    container.addEventListener("touchstart", onTouchStart, { passive: true });
    container.addEventListener("touchmove", onTouchMove, { passive: false });
    container.addEventListener("touchend", onTouchEnd);
    container.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    const resize = () => {
      const s = getSize();
      renderer.setSize(s.width, s.height, false);
      camera.aspect = s.width / s.height;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR));
    };
    window.addEventListener("resize", resize);
    resize();

    let rafId = 0;
    const animate = (time: number) => {
      rafId = requestAnimationFrame(animate);
      const cfg = configRef.current;
      const motionEnabled = !prefersReducedMotionRef.current;
      const rawDelta = lastFrameTime ? (time - lastFrameTime) / 1000 : 0.016;
      const deltaTime = Math.min(rawDelta, 0.1);
      lastFrameTime = time;
      
      if (motionEnabled && isScrolling) {
        scrollTarget += scrollMomentum;
        scrollMomentum *= MOMENTUM_FRICTION;
        if (Math.abs(scrollMomentum) < 0.001) scrollMomentum = 0;
      }
      
      if (cfg.autoplayEnabled && motionEnabled && !isScrolling && !isDragging) {
        if (isHovered) { autoplayIdleTime = 0; } 
        else {
          autoplayIdleTime += deltaTime;
          if (autoplayIdleTime > 0.8) scrollTarget += cfg.autoplaySpeed * deltaTime * 0.05;
        }
      } else { autoplayIdleTime = 0; }
      
      scrollPosition += (scrollTarget - scrollPosition) * SCROLL_SMOOTHING;
      
      if (Math.abs(scrollPosition) > loopLength * 1000) {
        const offset = Math.floor(scrollPosition / loopLength) * loopLength;
        scrollPosition -= offset; scrollTarget -= offset;
      }
      
      const isHoriz = cfg.direction === "horizontal";
      let closestDist = Infinity, closestIdx = 0;
      
      for (const mesh of meshes) {
        const { offset, index } = mesh.userData;
        let pos = -(offset - wrap(scrollPosition, loopLength));
        pos = wrap(pos + halfLoop, loopLength) - halfLoop;
        mesh.rotation.set(0, 0, 0); mesh.position.z = 0;
        
        if (mesh.userData.verticesBent && cfg.effectPreset !== "cylinder") {
          const flatPos = mesh.geometry.attributes.position;
          const orig = mesh.userData.originalVertices;
          for (let i = 0; i < flatPos.count; i++) flatPos.setZ(i, orig[i * 3 + 2]);
          flatPos.needsUpdate = true; mesh.geometry.computeVertexNormals();
          mesh.userData.verticesBent = false;
        }
        
        if (isHoriz) { mesh.position.x = pos; mesh.position.y = 0; } 
        else { mesh.position.y = pos; mesh.position.x = 0; }
        
        if (Math.abs(pos) < closestDist) { closestDist = Math.abs(pos); closestIdx = index; }
        
        if (cfg.activeScale > 1) {
          const t = Math.max(0, 1 - Math.abs(pos) / 1);
          const s = 1 + (cfg.activeScale - 1) * t * t * t;
          mesh.scale.set(s, s, 1);
        } else { mesh.scale.set(1, 1, 1); }
        
        const mat = mesh.material as THREE.ShaderMaterial;
        const target = mesh.userData.opacityTarget || 0;
        const loadRamp = mesh.userData.loadRamp ?? 0;
        if (loadRamp < target) mesh.userData.loadRamp = Math.min(target, loadRamp + deltaTime * 1.8);
        mat.uniforms.uOpacity.value = mesh.userData.loadRamp;
        
        if (Math.abs(pos) < halfLoop + (isHoriz ? cfg.slideHeightUnit * cfg.slideAspectRatio : cfg.slideHeightUnit)) {
          if (cfg.effectPreset === "cylinder") applyCylinderEffect(mesh, pos, cfg, isHoriz);
          else applyCoverflowEffect(mesh, pos, cfg, isHoriz);
        }
      }
      
      if (closestIdx !== activeSlideIndex) {
        activeSlideIndex = closestIdx; activeSlideIndexRef.current = closestIdx;
        startTransition(() => setActiveSlide(closestIdx));
      }
      renderer.render(scene, camera);
    };
    rafId = requestAnimationFrame(animate);
    
    return () => {
      disposed = true; cancelAnimationFrame(rafId);
      if (scrollTimeout) clearTimeout(scrollTimeout);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      container.removeEventListener("wheel", onWheel);
      container.removeEventListener("touchstart", onTouchStart);
      container.removeEventListener("touchmove", onTouchMove);
      container.removeEventListener("touchend", onTouchEnd);
      container.removeEventListener("pointerdown", onPointerDown);
      for (const mesh of meshes) {
        mesh.geometry.dispose();
        const mat = mesh.material as THREE.ShaderMaterial;
        if (mat.uniforms.uTexture?.value) mat.uniforms.uTexture.value.dispose();
        mat.dispose();
      }
      renderer.dispose();
      if (renderer.domElement.parentElement === container) container.removeChild(renderer.domElement);
    };
  }, [slides, direction]);

  return (
    <div style={{ ...style, width: "100%", height: "100%", position: "relative" }} ref={outerRef}>
      <div ref={canvasContainerRef} style={{ width: "100%", height: "100%", outline: "none", touchAction: "none" }} />
      {slides.length > 0 && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-col items-center pointer-events-none">
          <h3 className="text-white font-mono tracking-widest text-lg drop-shadow-md">{slides[activeSlide]?.title}</h3>
          <p className="text-white/70 text-sm">{slides[activeSlide]?.desc}</p>
        </div>
      )}
    </div>
  );
}
