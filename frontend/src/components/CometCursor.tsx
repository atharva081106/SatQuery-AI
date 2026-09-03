"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const FALLBACK_TRAIL = { r: 255, g: 93, b: 0, a: 1 };
const FALLBACK_CORE = { r: 255, g: 217, b: 184, a: 1 };

const PRESETS = {
  sharp: { blur: 0, length: 20, thickness: 9, headSize: 6, curve: 2, smoothing: 0.45, brightness: 1.1 },
  soft: { blur: 14, length: 28, thickness: 14, headSize: 10, curve: 3, smoothing: 0.32, brightness: 1 },
  nebula: { blur: 30, length: 46, thickness: 24, headSize: 18, curve: 4, smoothing: 0.2, brightness: 1.3 },
  ghost: { blur: 52, length: 64, thickness: 32, headSize: 22, curve: 6, smoothing: 0.12, brightness: 0.7 },
};

function parseColor(input: string, fallback: any) {
  if (typeof input !== "string" || input.trim() === "") return fallback;
  let value = input.trim();
  const token = value.match(/var\([^,]+,\s*(.+)\)\s*$/i);
  if (token) value = token[1].trim();
  if (value.charAt(0) === "#") {
    let hex = value.slice(1);
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    if (hex.length === 6 || hex.length === 8) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      const a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1;
      if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return fallback;
      return { r, g, b, a: a > 0 ? a : 1 };
    }
    return fallback;
  }
  const numbers = value.match(/[\d.]+/g);
  if (numbers && numbers.length >= 3) {
    const r = parseFloat(numbers[0]);
    const g = parseFloat(numbers[1]);
    const b = parseFloat(numbers[2]);
    const a = numbers.length > 3 ? parseFloat(numbers[3]) : 1;
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return fallback;
    return { r, g, b, a: a > 0 ? a : 1 };
  }
  return fallback;
}

function rgba(color: any, alpha: number) {
  const a = Math.max(0, Math.min(1, alpha * color.a));
  return `rgba(${color.r}, ${color.g}, ${color.b}, ${a})`;
}

function smoothPath(source: any[], iterations: number) {
  let path = source;
  for (let k = 0; k < iterations; k++) {
    if (path.length < 3) break;
    const out = [path[0]];
    for (let i = 1; i < path.length - 1; i++) {
      out.push({
        x: (path[i - 1].x + path[i].x * 2 + path[i + 1].x) / 4,
        y: (path[i - 1].y + path[i].y * 2 + path[i + 1].y) / 4,
      });
    }
    out.push(path[path.length - 1]);
    path = out;
  }
  return path;
}

interface CometCursorProps {
  variant?: "sharp" | "soft" | "nebula" | "ghost" | "custom";
  trailColor?: string;
  coreColor?: string;
  blur?: number;
  length?: number;
  thickness?: number;
  headSize?: number;
  curve?: number;
  smoothing?: number;
  brightness?: number;
  hideCursor?: boolean;
  disableOnTouch?: boolean;
  respectReducedMotion?: boolean;
  layer?: number;
}

export default function CometCursor(props: CometCursorProps) {
  const {
    variant = "soft",
    trailColor = "#FF5D00",
    coreColor = "#FFD9B8",
    blur = 14,
    length = 28,
    thickness = 14,
    headSize = 10,
    curve = 3,
    smoothing = 0.32,
    brightness = 1,
    hideCursor = false,
    disableOnTouch = true,
    respectReducedMotion = true,
    layer = 100,
  } = props;

  const glowRef = useRef<HTMLCanvasElement>(null);
  const coreRef = useRef<HTMLCanvasElement>(null);
  const settingsRef = useRef(PRESETS.soft);
  const [mounted, setMounted] = useState(false);

  const resolved =
    variant === "custom"
      ? { blur, length, thickness, headSize, curve, smoothing, brightness }
      : PRESETS[variant] || PRESETS.soft;

  settingsRef.current = resolved;

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    const glow = glowRef.current;
    const core = coreRef.current;
    if (!glow || !core) return;
    if (disableOnTouch && window.matchMedia("(pointer: coarse)").matches) return;
    if (respectReducedMotion && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const gx = glow.getContext("2d");
    const cx = core.getContext("2d");
    if (!gx || !cx) return;

    const trail = parseColor(trailColor, FALLBACK_TRAIL);
    const inner = parseColor(coreColor, FALLBACK_CORE);

    let width = 0;
    let height = 0;
    let frame = 0;
    let running = false;
    let idleFrames = 0;
    const points: any[] = [];
    const pos = { x: -400, y: -400 };
    const lead = { x: -400, y: -400 };
    const target = { x: -400, y: -400 };
    let started = false;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      for (const canvas of [glow, core]) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
      }
      gx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const clearCanvases = () => {
      gx.clearRect(0, 0, width, height);
      cx.clearRect(0, 0, width, height);
    };

    const paint = (ctx: CanvasRenderingContext2D, composite: string, color: any, widthMul: number, alphaMul: number, headMul: number, path: any[]) => {
      ctx.clearRect(0, 0, width, height);
      if (path.length < 3) return;
      const s = settingsRef.current;
      ctx.globalCompositeOperation = composite as GlobalCompositeOperation;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      const mid0 = { x: (path[0].x + path[1].x) / 2, y: (path[0].y + path[1].y) / 2 };
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      ctx.lineTo(mid0.x, mid0.y);
      ctx.lineWidth = Math.max(0.15, s.thickness * widthMul);
      ctx.strokeStyle = rgba(color, s.brightness * alphaMul);
      ctx.stroke();
      const n = path.length;
      for (let i = 1; i < n - 1; i++) {
        const t = 1 - i / n;
        const w = s.thickness * widthMul * t;
        if (w < 0.15) continue;
        const prev = path[i - 1];
        const current = path[i];
        const next = path[i + 1];
        const m1 = { x: (prev.x + current.x) / 2, y: (prev.y + current.y) / 2 };
        const m2 = { x: (current.x + next.x) / 2, y: (current.y + next.y) / 2 };
        ctx.beginPath();
        ctx.moveTo(m1.x, m1.y);
        ctx.quadraticCurveTo(current.x, current.y, m2.x, m2.y);
        ctx.lineWidth = w;
        ctx.strokeStyle = rgba(color, t * s.brightness * alphaMul);
        ctx.stroke();
      }
      const r = s.headSize * headMul;
      if (r > 0.6) {
        const head = path[0];
        const gradient = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, r);
        gradient.addColorStop(0, rgba(color, s.brightness * alphaMul));
        gradient.addColorStop(1, rgba(color, 0));
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(head.x, head.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const stop = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      running = false;
      points.length = 0;
      clearCanvases();
    };

    const loop = () => {
      const s = settingsRef.current;
      lead.x += (target.x - lead.x) * s.smoothing;
      lead.y += (target.y - lead.y) * s.smoothing;
      pos.x += (lead.x - pos.x) * s.smoothing;
      pos.y += (lead.y - pos.y) * s.smoothing;
      points.unshift({ x: pos.x, y: pos.y });
      while (points.length > s.length) points.pop();
      const path = smoothPath(points, s.curve);
      paint(gx, "lighter", trail, 1, 0.5, 1, path);
      paint(cx, "source-over", inner, 0.34, 1, 0.38, path);
      idleFrames += 1;
      const settled = Math.abs(target.x - pos.x) < 0.2 && Math.abs(target.y - pos.y) < 0.2;
      if (settled && idleFrames > s.length + 12) {
        stop();
        return;
      }
      frame = requestAnimationFrame(loop);
    };

    const start = () => {
      if (running || document.hidden) return;
      running = true;
      idleFrames = 0;
      frame = requestAnimationFrame(loop);
    };

    const onMove = (event: PointerEvent) => {
      target.x = event.clientX;
      target.y = event.clientY;
      if (!started) {
        started = true;
        pos.x = lead.x = event.clientX;
        pos.y = lead.y = event.clientY;
      }
      idleFrames = 0;
      start();
    };

    const onVisibility = () => {
      if (document.hidden) stop();
    };

    const onBlur = () => stop();

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onMove as any, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);

    const previousCursor = document.body.style.cursor;
    if (hideCursor) document.body.style.cursor = "none";

    return () => {
      stop();
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onMove as any);
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
      document.body.style.cursor = previousCursor;
    };
  }, [mounted, trailColor, coreColor, hideCursor, disableOnTouch, respectReducedMotion]);

  if (!mounted || typeof document === "undefined") return null;

  const canvasStyle: React.CSSProperties = {
    position: "fixed",
    top: 0,
    left: 0,
    display: "block",
    pointerEvents: "none",
  };

  return createPortal(
    <div style={{ position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh", pointerEvents: "none", zIndex: Number(layer) || 100 }}>
      <canvas ref={glowRef} style={{ ...canvasStyle, filter: resolved.blur > 0 ? `blur(${resolved.blur}px)` : "none" }} />
      <canvas ref={coreRef} style={canvasStyle} />
    </div>,
    document.body
  );
}
