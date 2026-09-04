import { useEffect, useRef, useState } from 'react';
import * as GeoTIFF from 'geotiff';

interface TiffPreviewProps {
  file: File;
  onClick?: () => void;
  className?: string;
}

export default function TiffPreview({ file, onClick, className = "" }: TiffPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    async function renderTiff() {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
        const image = await tiff.getImage();
        
        const rasters = await image.readRasters();
        const width = image.getWidth();
        const height = image.getHeight();
        
        if (!active) return;
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        // Scale down for preview if very large
        const maxDim = 256;
        const scale = Math.min(1, maxDim / Math.max(width, height));
        canvas.width = Math.floor(width * scale);
        canvas.height = Math.floor(height * scale);
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Create an ImageData object for offscreen
        const imgData = new ImageData(width, height);
        const data = imgData.data;
        
        if (rasters.length === 1) {
          const r0 = rasters[0] as any; // Float32Array, Uint8Array or Uint16Array
          for (let i = 0; i < width * height; i++) {
             let val = r0[i];
             if (val > 255) val = val >> 8; // Naive scaling down for 16-bit
             if (val < 0) val = 0;
             if (val > 255) val = 255;
             data[i*4] = val;
             data[i*4+1] = val;
             data[i*4+2] = val;
             data[i*4+3] = 255;
          }
        } else if (rasters.length >= 3) {
          const r0 = rasters[0] as any;
          const r1 = rasters[1] as any;
          const r2 = rasters[2] as any;
          for (let i = 0; i < width * height; i++) {
             let valR = r0[i];
             let valG = r1[i];
             let valB = r2[i];
             if (valR > 255) valR = valR >> 8;
             if (valG > 255) valG = valG >> 8;
             if (valB > 255) valB = valB >> 8;
             data[i*4] = valR;
             data[i*4+1] = valG;
             data[i*4+2] = valB;
             data[i*4+3] = 255;
          }
        }
        
        // Draw to offscreen first
        const offscreen = document.createElement('canvas');
        offscreen.width = width;
        offscreen.height = height;
        const offCtx = offscreen.getContext('2d');
        if (offCtx) {
          offCtx.putImageData(imgData, 0, 0);
          ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvas.width, canvas.height);
        }
        
      } catch (err) {
        console.error("Failed to render TIFF", err);
        setError(true);
      }
    }
    renderTiff();
    return () => { active = false; };
  }, [file]);
  
  if (error) {
    return (
      <div 
        className={`bg-[#222] flex items-center justify-center text-[10px] text-white/50 cursor-pointer ${className}`}
        onClick={onClick}
      >
        TIF
      </div>
    );
  }

  return (
    <canvas 
      ref={canvasRef} 
      className={`object-cover cursor-pointer hover:opacity-70 transition-opacity bg-black ${className}`}
      onClick={onClick}
    />
  );
}
