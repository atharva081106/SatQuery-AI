"use client";

import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

interface CursorDotTrailProps {
  color?: string;
  colorInverted?: string;
  size?: number;
  hoverSize?: number;
  borderWidth?: number;
  spring?: number;
  friction?: number;
  trailDuration?: number;
  transitionSpeed?: number;
  hideCursor?: boolean;
}

export default function CursorDotTrail(props: CursorDotTrailProps) {
  const {
    color = "#00f0ff",
    colorInverted = "#ffffff",
    size = 12,
    hoverSize = 40,
    borderWidth = 2,
    spring = 0.15,
    friction = 0.5,
    trailDuration = 200,
    transitionSpeed = 0.15,
    hideCursor = true,
  } = props;

  const isBrowser = typeof window !== "undefined" && typeof document !== "undefined";
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointsRef = useRef<Array<{ x: number; y: number; age: number }>>([]);
  const ballRef = useRef({ x: 0, y: 0 });
  const targetRef = useRef({ x: 0, y: 0 });
  const velocityRef = useRef({ x: 0, y: 0 });
  const animRef = useRef<number>();
  const lastTimeRef = useRef(0);
  
  const radiusRef = useRef(size / 2);
  const fillOpacityRef = useRef(1);
  const strokeOpacityRef = useRef(0);
  const lineOpacityRef = useRef(0);
  const lineWidthRef = useRef(0);
  const lineTargetWidthRef = useRef(0);
  const lineProgressRef = useRef(0);
  const lineYRef = useRef(0);

  useEffect(() => {
    if (!isBrowser) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (hideCursor) document.body.style.cursor = "none";
    const previousCursor = document.body.style.cursor;

    const resize = () => {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      
      const centerX = w / 2;
      const centerY = h / 2;
      if (!ballRef.current.x && !ballRef.current.y) {
        ballRef.current = { x: centerX, y: centerY };
      }
      if (!targetRef.current.x && !targetRef.current.y) {
        targetRef.current = { x: centerX, y: centerY };
      }
    };
    
    resize();
    
    const onMouseMove = (e: MouseEvent) => {
      targetRef.current = { x: e.clientX, y: e.clientY };
    };
    
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("resize", resize);
    
    const lerp = (from: number, to: number, amt: number) => from + (to - from) * amt;
    
    const animate = () => {
      const now = performance.now();
      const dt = Math.min(now - lastTimeRef.current, 33);
      lastTimeRef.current = now;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // physics
      const dx = targetRef.current.x - ballRef.current.x;
      const dy = targetRef.current.y - ballRef.current.y;
      velocityRef.current.x += dx * spring;
      velocityRef.current.y += dy * spring;
      velocityRef.current.x *= friction;
      velocityRef.current.y *= friction;
      ballRef.current.x += velocityRef.current.x;
      ballRef.current.y += velocityRef.current.y;
      
      // trail
      pointsRef.current.push({ x: ballRef.current.x, y: ballRef.current.y, age: 0 });
      for (let i = 0; i < pointsRef.current.length; i++) {
        pointsRef.current[i].age += dt;
      }
      pointsRef.current = pointsRef.current.filter((p) => p.age < trailDuration);
      
      // element under cursor
      const el = document.elementFromPoint(targetRef.current.x, targetRef.current.y);
      
      // hide-trail
      const isHideTrail = !!el?.closest('[aria-label~="trail{hide}"],[data-framer-name~="trail{hide}"]');
      if (isHideTrail) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        animRef.current = requestAnimationFrame(animate);
        return;
      }
      
      // invert color
      const isInvertColor = !!el?.closest('[aria-label~="trail{invert-color}"],[data-framer-name~="trail{invert-color}"]');
      const currentColor = isInvertColor ? colorInverted : color;
      
      if (pointsRef.current.length > 1) {
        ctx.beginPath();
        ctx.moveTo(pointsRef.current[0].x, pointsRef.current[0].y);
        for (let i = 1; i < pointsRef.current.length; i++) {
          const p = pointsRef.current[i];
          ctx.lineTo(p.x, p.y);
        }
        
        const oldest = pointsRef.current[0];
        const newest = pointsRef.current[pointsRef.current.length - 1];
        const oldestOpacity = 1 - oldest.age / trailDuration;
        
        const gradient = ctx.createLinearGradient(oldest.x, oldest.y, newest.x, newest.y);
        gradient.addColorStop(0, hexToRgba(currentColor, Math.max(0, oldestOpacity * 0.3)));
        gradient.addColorStop(1, hexToRgba(currentColor, 1));
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = Math.max(2, size / 4);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();
      }
      
      // interaction
      const isRing = !!el?.closest("a,button,[role~='button']");
      const hideDot = !!el?.closest('[aria-label~="button"]');
      const linkEl = el?.closest('[aria-label~="trail{link}"],[data-framer-name~="trail{link}"]');
      const workCardEl = el?.closest('[aria-label~="work-card"]');
      
      const isLink = !!linkEl;
      const isWorkCard = !!workCardEl;
      
      // link underline
      if (isLink && linkEl) {
        const rect = linkEl.getBoundingClientRect();
        ballRef.current.x = rect.left + rect.width / 2;
        ballRef.current.y = rect.bottom;
        lineTargetWidthRef.current = rect.width;
        lineYRef.current = rect.bottom;
      } else {
        lineTargetWidthRef.current = 0;
      }
      
      // radius / opacities
      const targetRadius = isWorkCard || (!isRing && !isLink) ? size / 2 : hoverSize / 2;
      radiusRef.current = lerp(radiusRef.current, targetRadius, transitionSpeed);
      
      fillOpacityRef.current = lerp(fillOpacityRef.current, isWorkCard ? 1 : (isRing || isLink) ? 0 : 1, transitionSpeed);
      strokeOpacityRef.current = lerp(strokeOpacityRef.current, isWorkCard ? 0 : isRing ? 1 : 0, transitionSpeed);
      lineOpacityRef.current = lerp(lineOpacityRef.current, isWorkCard ? 0 : isLink ? 1 : 0, transitionSpeed);
      
      // underline animation
      const targetProgress = isLink ? 1 : 0;
      lineProgressRef.current = lerp(lineProgressRef.current, targetProgress, 0.15);
      lineWidthRef.current = lerp(lineWidthRef.current, lineTargetWidthRef.current * lineProgressRef.current, transitionSpeed);
      
      if (!hideDot) {
        if (isLink && lineOpacityRef.current > 0.01 && linkEl) {
          const linkRect = linkEl.getBoundingClientRect();
          const y = linkRect.bottom + 1;
          const currentWidth = lineWidthRef.current;
          const startX = linkRect.left + (linkRect.width - currentWidth) / 2;
          
          ctx.beginPath();
          ctx.moveTo(startX, y);
          ctx.lineTo(startX + currentWidth, y);
          ctx.strokeStyle = hexToRgba(currentColor, lineOpacityRef.current);
          ctx.lineWidth = Math.max(1, borderWidth - 1);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(ballRef.current.x, ballRef.current.y, radiusRef.current, 0, Math.PI * 2);
          
          if (strokeOpacityRef.current > 0.01) {
            ctx.strokeStyle = hexToRgba(currentColor, strokeOpacityRef.current);
            ctx.lineWidth = borderWidth;
            ctx.stroke();
          }
          
          if (fillOpacityRef.current > 0.01) {
            ctx.fillStyle = hexToRgba(currentColor, fillOpacityRef.current);
            ctx.fill();
          }
        }
      }
      
      animRef.current = requestAnimationFrame(animate);
    };
    
    lastTimeRef.current = performance.now();
    animRef.current = requestAnimationFrame(animate);
    
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", resize);
      if (animRef.current) cancelAnimationFrame(animRef.current);
      if (hideCursor) document.body.style.cursor = previousCursor;
    };
  }, [color, colorInverted, size, hoverSize, borderWidth, spring, friction, trailDuration, transitionSpeed, hideCursor, isBrowser]);

  if (!isBrowser) {
    return null;
  }

  return createPortal(
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100vw",
        height: "100vh",
        display: "block",
        pointerEvents: "none",
        zIndex: 99999,
      }}
    />,
    document.body
  );
}

function hexToRgba(color: string, alpha: number) {
  if (!color) return `rgba(0,0,0,${alpha})`;
  
  if (color.startsWith("rgba")) {
    return color.replace(/rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*[^)]+\)/, (_, r, g, b) => `rgba(${r},${g},${b},${alpha})`);
  }
  
  if (color.startsWith("rgb(")) {
    const body = color.slice(4, -1);
    return `rgba(${body},${alpha})`;
  }
  
  if (color.startsWith("#")) {
    let r = 0, g = 0, b = 0;
    if (color.length === 4) {
      r = parseInt(color[1] + color[1], 16);
      g = parseInt(color[2] + color[2], 16);
      b = parseInt(color[3] + color[3], 16);
    } else if (color.length === 7 || color.length === 9) {
      r = parseInt(color.slice(1, 3), 16);
      g = parseInt(color.slice(3, 5), 16);
      b = parseInt(color.slice(5, 7), 16);
    }
    return `rgba(${r},${g},${b},${alpha})`;
  }
  
  return `rgba(0,0,0,${alpha})`;
}
