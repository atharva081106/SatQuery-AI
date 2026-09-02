"use client";

import { useState, useEffect } from "react";

const images = [
  "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?q=80&w=2072&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?q=80&w=2074&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1541873676-a18131494184?q=80&w=2018&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop"
];

export default function BackgroundSlideshow({ overlayOpacity = "bg-black/30", extraClasses = "" }: { overlayOpacity?: string, extraClasses?: string }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, 6000); // Change image every 6 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      {images.map((src, idx) => (
        <div
          key={idx}
          className={`absolute inset-0 z-0 bg-center bg-cover bg-no-repeat transition-opacity duration-1000 ${
            idx === currentIndex ? "opacity-100" : "opacity-0"
          } ${extraClasses}`}
          style={{ backgroundImage: `url('${src}')` }}
        />
      ))}
      <div className={`absolute inset-0 z-0 ${overlayOpacity}`}></div>
    </>
  );
}
