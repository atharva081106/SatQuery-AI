"use client";

import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { motion, useMotionValue, useSpring, useInView, animate, useTransform } from "framer-motion";

const SNAP_TIMEOUT_MS = 150;
const ITEM_ASPECT_RATIO = 16 / 9;
const ITEM_HEIGHT_PERCENT = 0.8; // Make slightly shorter so it fits nicely
const ITEM_WIDTH_PERCENT = 0.8;

interface MotionGalleryProps {
  images: { src: string; title: string; desc: string }[];
  itemWidth?: number; // scale factor
  gap?: number;
  padding?: number;
  borderRadius?: number;
  autoCentered?: boolean;
  scrollSpeed?: number;
  scrollSmoothness?: number;
}

export default function MotionGallery({
  images = [],
  itemWidth = 100,
  gap = 16,
  padding = 16,
  borderRadius = 12,
  autoCentered = true,
  scrollSpeed = 0.5,
  scrollSmoothness = 0.5,
}: MotionGalleryProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, amount: 0.2 });
  
  const [containerSize, setContainerSize] = useState({ width: 600, height: 400 });
  const [hasEntered, setHasEntered] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const snapTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (images.length > 0) {
      baseOffset.set(0);
      setCurrentIndex(0);
    }
  }, [images.length]);

  useEffect(() => {
    if (isInView && !hasEntered) setHasEntered(true);
  }, [isInView, hasEntered]);

  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      setContainerSize({
        width: Math.max(rect.width, 5),
        height: Math.max(rect.height, 5),
      });
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const baseItemHeight = containerSize.height * ITEM_HEIGHT_PERCENT;
  const baseItemWidth = baseItemHeight * ITEM_ASPECT_RATIO;
  
  const scaleFactor = itemWidth / 100;
  const finalItemWidth = baseItemWidth * scaleFactor;
  const itemHeight = baseItemHeight;
  const itemSize = Math.max(finalItemWidth + gap, 1);
  
  const baseOffset = useMotionValue(0);
  const springConfig = useMemo(
    () => ({
      stiffness: 500 - scrollSmoothness * 450,
      damping: 15 + scrollSmoothness * 25,
      mass: 0.6 + scrollSmoothness * 0.4,
    }),
    [scrollSmoothness]
  );
  const springOffset = useSpring(baseOffset, springConfig);

  useEffect(() => {
    if (!itemSize) return;
    unsubscribeRef.current?.();
    unsubscribeRef.current = springOffset.on("change", (v) => {
      if (images.length === 0) return;
      const pos = -v / itemSize;
      const idx = Math.round(pos);
      const newIndex = ((idx % images.length) + images.length) % images.length;
      setCurrentIndex(newIndex);
    });
    return () => {
      unsubscribeRef.current?.();
      unsubscribeRef.current = null;
    };
  }, [springOffset, images.length, itemSize]);

  const snapToNearest = () => {
    if (itemSize < 2 || images.length <= 1) return;
    const target = -Math.round(-baseOffset.get() / itemSize) * itemSize;
    animate(baseOffset, target, { type: "spring", ...springConfig });
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (images.length <= 1 || itemSize < 2) return;
    const delta = e.deltaX || e.deltaY;
    baseOffset.set(baseOffset.get() - delta * scrollSpeed);
    
    if (!autoCentered) return;
    if (snapTimeoutRef.current) clearTimeout(snapTimeoutRef.current);
    snapTimeoutRef.current = setTimeout(snapToNearest, SNAP_TIMEOUT_MS);
  };

  // Touch handling
  const touchStartRef = useRef<{x: number, y: number} | null>(null);
  const touchStartOffsetRef = useRef(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onTouchStart = (e: TouchEvent) => {
      if (images.length <= 1 || itemSize < 2) return;
      const touch = e.touches[0];
      touchStartRef.current = { x: touch.clientX, y: touch.clientY };
      touchStartOffsetRef.current = baseOffset.get();
    };

    const onTouchMove = (e: TouchEvent) => {
      if (!touchStartRef.current || images.length <= 1 || itemSize < 2) return;
      const touch = e.touches[0];
      const deltaX = touch.clientX - touchStartRef.current.x;
      baseOffset.set(touchStartOffsetRef.current + deltaX);
    };

    const onTouchEnd = () => {
      touchStartRef.current = null;
      if (autoCentered && images.length > 1 && itemSize >= 2) {
        snapToNearest();
      }
    };

    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd, { passive: true });
    el.addEventListener("touchcancel", onTouchEnd, { passive: true });

    return () => {
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
      el.removeEventListener("touchcancel", onTouchEnd);
    };
  }, [images.length, itemSize, autoCentered, baseOffset, springConfig]);

  const goToIndex = (index: number) => {
    if (images.length <= 1 || itemSize < 2) return;
    const current = Math.round(-springOffset.get() / itemSize);
    const wrapped = ((current % images.length) + images.length) % images.length;
    let diff = index - wrapped;
    if (diff > images.length / 2) diff -= images.length;
    if (diff < -images.length / 2) diff += images.length;
    animate(baseOffset, springOffset.get() - diff * itemSize, {
      type: "spring",
      ...springConfig,
    });
  };

  if (!images.length) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%" }}>
      <motion.div
        ref={containerRef}
        initial={{ opacity: 0, scale: 0.96 }}
        animate={hasEntered ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.6, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          overflow: "hidden",
          touchAction: "none",
        }}
        onWheel={handleWheel}
      >
        <div style={{ width: "100%", height: "100%", position: "relative" }}>
          {images.map((img, i) => (
            <GalleryItem
              key={`${i}-${img.src}`}
              itemIndex={i}
              image={img}
              itemWidth={finalItemWidth}
              itemHeight={itemHeight}
              itemSize={itemSize}
              borderRadius={borderRadius}
              containerSize={containerSize}
              springOffset={springOffset}
              hasEntered={hasEntered}
              totalImages={images.length}
            />
          ))}
        </div>
      </motion.div>
    </div>
  );
}

function GalleryItem({
  itemIndex,
  image,
  itemWidth,
  itemHeight,
  itemSize,
  borderRadius,
  containerSize,
  springOffset,
  hasEntered,
  totalImages,
}: any) {
  const getPosition = (offset: number) => {
    const total = totalImages * itemSize;
    if (!total) return 0;
    let pos = itemIndex * itemSize + offset;
    pos = ((pos % total) + total) % total;
    if (pos > total / 2) pos -= total;
    return pos;
  };

  const x = useTransform(
    springOffset,
    (v: number) => containerSize.width / 2 - itemWidth / 2 + getPosition(v)
  );
  
  const y = containerSize.height / 2 - itemHeight / 2;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={hasEntered ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.5, delay: 0.1 + itemIndex * 0.03, ease: [0.25, 0.46, 0.45, 0.94] }}
      style={{
        width: itemWidth,
        height: itemHeight,
        borderRadius,
        overflow: "hidden",
        position: "absolute",
        x,
        y,
        top: 0,
        left: 0,
        willChange: "transform",
      }}
      className="group relative cursor-pointer border border-white/20"
      onClick={() => window.open(image.src, "_blank")}
    >
      <img
        src={image.src}
        alt={image.title || ""}
        style={{ width: "100%", height: "100%", objectFit: "cover", pointerEvents: "none", userSelect: "none" }}
        draggable={false}
      />
      {/* Title Overlay overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-6 pointer-events-none">
        <span className="text-[#00F0FF] font-mono text-sm tracking-widest mb-2">{image.title}</span>
        <span className="text-white/80 font-sans text-xs">{image.desc}</span>
      </div>
    </motion.div>
  );
}
