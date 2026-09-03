# SatQuery AI: 3-Minute Prototype Walkthrough Script

*This script is designed for a fast-paced, high-impact 2-3 minute live demo. It focuses on the core problem, the solution, and the "wow" factor of the live interface.*

---

## Part 1: The Hook (0:00 - 0:30)

**(Screen: Homepage with the 3D rotating globe)**

**Speaker:** "Good morning judges. Analyzing satellite imagery has traditionally been a bottleneck. It requires complex desktop GIS software, massive downloads, and specialized training. 

Today, we are presenting **SatQuery AI**—a platform that completely democratizes earth observation. We’ve built an aerospace-grade, natural language interface that allows anyone to query and analyze satellite data just by chatting with an AI. 

Let me show you how it works."

---

## Part 2: Acquisition & The Map Explorer (0:30 - 1:00)

**(Screen: Click "Enter Command Center" and open the Map Explorer Modal)**

**Speaker:** "We start in the Map Explorer. I don’t need to download massive raster files. I just draw a bounding box over my area of interest—let’s say a coastal region here. 

When I click 'Acquire Data', our backend immediately communicates with the Copernicus Data Space Ecosystem to pull live, high-resolution Sentinel satellite data. 

**(Screen: Map Explorer closes, the image drops into the chat interface)**

And just like that, the imagery is loaded directly into our multimodal chat session."

---

## Part 3: The AI Agents in Action (1:00 - 2:00)

**(Screen: Type a query into the chat bar, e.g., "Highlight the water bodies and vegetation")**

**Speaker:** "Now for the actual analysis. I’m going to ask the system to highlight the water bodies in this region. 

*(Hit enter/submit)*

Behind the scenes, we aren't just sending this to a generic ChatGPT wrapper. We built a custom **Multi-Agent Orchestrator**. The system reads my prompt, understands my intent, and routes it to our specialized Visual Grounding engine.

**(Screen: The AI responds with the bounded image and GeoJSON metadata)**

As you can see, it instantly isolated the water bodies and drew precise bounding boxes around them. But it doesn't stop at just pretty pictures. It generates a synthetic Grad-CAM heatmap so we know exactly where the AI is focusing, making the AI completely explainable."

---

## Part 4: The "Wow" Factor - Bi-Temporal Analysis (2:00 - 2:30)

**(Screen: Click the new 'Acquire Map' icon in the chat bar. Draw a new box, acquire a second image. Then type: "What has changed between these images?")**

**Speaker:** "What if we want to detect changes over time? I can seamlessly acquire a second image directly from the chat interface. I’ll ask: *'What has changed?'*

Our Agentic Controller recognizes the two images and the temporal query, and routes it to our Bi-Temporal Change Agent. 

**(Screen: AI returns the Swipe Slider and the change-detected image)**

The system automatically verifies spatial alignment, runs absolute differencing, and highlights structural changes in red. We’ve even included an interactive Swipe Slider so analysts can manually verify the sub-pixel changes."

---

## Part 5: The GIS Handoff & Conclusion (2:30 - 3:00)

**(Screen: Click the "Download GeoJSON Vector" button)**

**Speaker:** "Finally, a pretty chat interface isn't enough for real-world deployment. Every piece of analysis, every bounding box, and every detected change is mapped back into real geographic coordinates. 

With one click, I can download this **GeoJSON FeatureCollection**. This file can be immediately handed off to a disaster response team and imported into their existing military or civic GIS infrastructure like ArcGIS.

SatQuery AI bridges the gap between raw space data and actionable intelligence—accessible to anyone, in seconds. 

Thank you, we'd love to take your questions."
