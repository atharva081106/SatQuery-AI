# SatQuery AI: Official PPT Slide-by-Slide Script

*This script is designed to accompany a standard 8–10 slide hackathon pitch deck. Each slide section includes what should be visually on the slide and exactly what you should say.*

---

## Slide 1: Title Slide
**Visual:** Project Name (SatQuery AI), Logo, Team Name, and a sleek, dark-themed satellite or globe background.
**Speaker:** 
"Good morning, judges. We are [Team Name], and today we are thrilled to present **SatQuery AI**—a multimodal, agentic platform that is democratizing Earth observation and satellite imagery analysis."

---

## Slide 2: The Problem
**Visual:** Three bullet points emphasizing complexity, cost, and high barriers to entry. Maybe an image of an overly complex GIS interface (like QGIS) looking confusing.
**Speaker:** 
"Currently, unlocking the power of satellite imagery is incredibly difficult. 
First, acquiring high-quality imagery requires navigating clunky databases. Second, analyzing that data—whether it's detecting changes over time or highlighting infrastructure—requires downloading massive datasets and using complex, expensive GIS software. 
This creates a massive barrier to entry for first responders, journalists, and smaller organizations who need actionable intelligence immediately."

---

## Slide 3: Our Solution
**Visual:** A clean mockup of the SatQuery AI chat interface. Three key pillars: Natural Language, AI Agents, Instant GIS Export.
**Speaker:** 
"Our solution is **SatQuery AI**. We have eliminated the learning curve entirely. We built a platform where analyzing satellite data is as easy as sending a text message. 
You simply draw a box on a map, pull live satellite data, and chat with an AI that instantly understands the imagery, extracts features, and even tracks changes over time—all directly in your browser."

---

## Slide 4: System Architecture (High-Level)
**Visual:** A simple flowchart: User Interface ➔ Agentic Controller ➔ Copernicus Data Space Ecosystem (CDSE) + Specialist AI Models ➔ GeoJSON Output.
**Speaker:** 
"Under the hood, we aren't just using a basic chatbot. We built a sophisticated **Multi-Agent Orchestrator**. 
When a user submits a query, our backend talks directly to the Copernicus Data Space API to fetch live Sentinel data. Then, our Agentic Controller parses the user's intent and routes it to one of our specialized machine learning models—whether that's a Vision-Language model for scene captioning, or a Computer Vision agent for pixel-perfect change detection."

---

## Slide 5: Key Features - Visual Grounding & Bi-Temporal Change
**Visual:** Split screen. Left side: Bounding boxes around water/vegetation. Right side: Before & After satellite images with red bounding boxes showing changes.
**Speaker:** 
"Two of our most powerful capabilities are Visual Grounding and Bi-Temporal Analysis. 
If a user asks to 'find water bodies', our system isolates those features and draws exact bounding boxes around them. 
If a user uploads two images from different dates, our system automatically co-registers the images, analyzes the structural differences, and highlights exactly what infrastructure or terrain has changed between those two dates."

---

## Slide 6: The "Hackathon Flex" (Resource Optimization)
**Visual:** A graphic showing a heavy ML model crossing out to an optimized OpenCV icon. Text: "100% Uptime, Low-Resource Fallback."
**Speaker:** 
"One of the biggest challenges with AI is cost and server crashing due to memory limits. We engineered our backend to be production-ready and highly resilient. 
While we utilize advanced HuggingFace Foundation Models when GPU power is available, we built a proprietary, low-resource **Heuristic Engine** using OpenCV. If the server runs low on RAM, the system seamlessly falls back to this engine, ensuring our users get lightning-fast analysis with 100% uptime, even on free-tier infrastructure."

---

## Slide 7: Real-World Application & GIS Interoperability
**Visual:** A screenshot of a GeoJSON file being loaded into standard GIS software (like ArcGIS or QGIS).
**Speaker:** 
"But a pretty chat interface isn't enough for military or disaster response teams. That's why we made SatQuery AI completely interoperable. 
Every bounding box, heatmap, and detected change our AI generates is translated back into actual Geographic Coordinates (Latitude and Longitude). With a single click, users can export this as a **GeoJSON FeatureCollection** and plug it directly into their existing professional workflows."

---

## Slide 8: Future Roadmap
**Visual:** Icons representing new features (e.g., Mobile App, Drone Integration, More Satellite Constellations).
**Speaker:** 
"Looking forward, we plan to integrate higher-resolution commercial satellites, add support for drone imagery, and introduce predictive analytics to forecast environmental changes before they happen."

---

## Slide 9: Conclusion & Q&A
**Visual:** "Thank You", Team Names, GitHub Link, or QR code to the live demo.
**Speaker:** 
"SatQuery AI bridges the gap between raw space data and actionable intelligence, making it accessible to anyone, in seconds. 
Thank you for your time. We would now love to show you a live demo or take any questions you might have."
