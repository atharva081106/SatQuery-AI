"use client";

import { useState } from 'react';

const faqs = [
  {
    question: "WHAT IS THE DEFINED INPUT SCOPE?",
    answer: "Our pipeline is optimized for three specific configurations:\n\n• Single image: One optical/multispectral or SAR image for captioning, visual question answering, and text-guided region grounding.\n\n• Cross-modal pair: Co-registered optical/multispectral and SAR images of the same geographic area for joint information extraction and cross-modal analysis.\n\n• Bi-temporal pair: Two spatially corresponding images of the same geographic area acquired at different times for change detection, change description, and change-based visual question answering."
  },
  {
    question: "WHAT IMAGERY FORMATS ARE SUPPORTED?",
    answer: "GeoTIFF or TIFF for geospatial imagery. PNG and JPEG inputs may be accepted only for the prescribed public benchmark datasets."
  },
  {
    question: "HOW DOES MULTIMODAL REASONING WORK?",
    answer: "Our pipeline fuses spatial feature extraction with a large language model. It interprets your natural language query, correlates it directly with the pixel-level features in the uploaded imagery, and generates an exact, grounded response."
  },
  {
    question: "WHAT IS THE INFERENCE LATENCY?",
    answer: "Our optimized architecture delivers inference in under 1.5 seconds for standard 1024px tiles. It is built for real-time tactical edge deployments."
  }
];

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="w-full max-w-4xl flex flex-col">
      {faqs.map((faq, index) => {
        const isOpen = openIndex === index;
        return (
          <div 
            key={index} 
            className="border-b border-[#2a2a2f] overflow-hidden"
          >
            <button 
              onClick={() => toggle(index)}
              className="w-full py-6 flex justify-between items-center text-left hover:text-white/80 transition-colors focus:outline-none"
            >
              <span className="text-sm tracking-widest uppercase font-semibold">{faq.question}</span>
              <span className="text-xl font-light text-white/50">{isOpen ? '−' : '+'}</span>
            </button>
            <div 
              className={`transition-all duration-300 ease-in-out ${isOpen ? 'max-h-80 pb-6 opacity-100' : 'max-h-0 opacity-0'}`}
            >
              <p className="text-white/60 text-sm leading-relaxed max-w-3xl whitespace-pre-wrap">
                {faq.answer}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
