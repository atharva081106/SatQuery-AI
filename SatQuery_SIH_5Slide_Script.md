# SatQuery AI: Smart India Hackathon (SIH) Official Pitch Script
**Problem Statement ID:** 26167 | **Organization:** ISRO (Dept. of Space)

*This script is meticulously aligned with the official ISRO problem statement requirements, hitting every mandatory functional scope, evaluation metric, and technical capability requested.*

---

## Slide 1: The Foundation (Problem & Solution)
*Visuals: Bullet points of the Problem Statement, Market Gap, Our Solution, USP, and Key SIH Requirements.*

**Speaker:** 
"Good morning, judges. We are presenting **SatQuery AI** for ISRO's Problem Statement 26167. 

**[Problem Statement & Market Gap]**
Currently, remote-sensing AI solutions are highly fragmented. They are isolated applications built for single, predefined tasks—like just land-cover classification or just object detection. They require users to deeply understand GIS workflows, sensor characteristics, and model selection. Consequently, non-expert users, like first responders or urban planners, cannot easily extract meaningful intelligence using simple natural language. 

**[Our Solution & USP]**
Our solution, SatQuery AI, bridges this gap. It is an interactive, agentic vision-language assistant for multimodal remote sensing. Our **Unique Selling Proposition** is our **Query-Driven Agentic Framework**. We do not rely on a single, generic ChatGPT wrapper—which we know fails on top-down satellite imagery. Instead, our system acts as an orchestrator that dynamically selects, sequences, and executes specialized remote-sensing models based on the user's specific query.

**[Key Features & SIH Requirements]**
We have successfully implemented every mandatory SIH requirement:
1. A Remote-Sensing adapted Vision-Language baseline.
2. Single-image VQA and text-guided region grounding.
3. Multi-image change analysis (Bi-temporal).
4. Cross-modal analysis (combining Optical and SAR data).
5. Full Agentic Orchestration."

---

## Slide 2: Technical Approach & Live Walkthrough
*Visuals: Architecture diagram showing the Agentic Controller, Registry, and Output Integration, transitioning to a live application demo.*

**Speaker:** 
"Let's look at our **Technical Approach** and a live walkthrough.

When a user uploads GeoTIFFs or benchmark JPEGs and types a query, our **Agentic Controller** takes over. It interprets the query, validates the input modality and metadata, and selects the appropriate model from our predefined registry. 

**(Live Walkthrough Starts)**
*   **Single-Image Grounding:** I upload an optical image and ask, *'Highlight the water body referred to in the query.'* The controller routes this to our grounding model, which isolates the specific pixels and draws a precise bounding box.
*   **Cross-Modal Analysis:** Single optical images are often blocked by clouds. I upload an Optical image and a Synthetic Aperture Radar (SAR) image—which penetrates clouds—and ask it to *'identify built-up regions'*. The agent mathematically fuses the spectral data of the optical image with the structural data of the SAR image to output a highly reliable answer.
*   **Bi-Temporal Analysis:** I upload two spatially corresponding images from different dates. I ask, *'Has the built-up area increased, decreased, or remained unchanged?'* The controller routes to our Change-VQA tool, runs spatial co-registration checks, and highlights the exact structural differences.
*   **Execution Summary:** Finally, for every query, the system generates an auditable execution trace—showing the selected task, the model provenance, and visual evidence."

---

## Slide 3: Feasibility & Viability
*Visuals: Technical architecture flow, API integrations, and resource optimization.*

**Speaker:** 
"Regarding **Feasibility and Viability**:

**[Technical Feasibility]**
We engineered the backend to handle the exact input scope requested by ISRO. Our pipeline flawlessly processes standard GeoTIFF formats and handles spatial bounding box metadata for Cartosat-2S and RISAT SAR pairs. Our Agentic Controller strictly confines itself to permitted task parameters, meaning it is highly stable and resistant to AI hallucinations. 

**[Economic Viability]**
Because our orchestrator routes to lightweight, specialized heuristic tools for tasks like change detection—rather than forcing every image through massive, computationally expensive multimodal LLMs—our operational costs and server loads are drastically reduced. This makes it highly viable for large-scale deployment across government departments."

---

## Slide 4: Impact and Benefits
*Visuals: Traditional vs. SatQuery workflow diagram, XAI/Audit Report screenshots, Sectors impacted.*

**Speaker:** 
"The impact of SatQuery AI is transformative.

**[Traditional vs. Our Workflow]**
The traditional workflow requires a user to manually select a model, pre-process the SAR/Optical data, and run isolated scripts. With SatQuery, the user simply asks a question in English. The agent handles the modality checks and model sequencing autonomously.

**[Explainable AI, Trust & Sovereignty]**
In sectors like defense and disaster management, trust is paramount. Our system provides **Explainable AI** by returning visual evidence (like Grad-CAM heatmaps or bounding boxes) alongside textual answers. Furthermore, the mandatory **Auditable Execution Summary** ensures complete transparency over exactly which model generated the intelligence, ensuring strict data sovereignty.

**[Sector Impact]**
By providing robust Day-and-Night capability through Cross-Modal SAR fusion, we are enabling continuous, 24/7 impact across agricultural monitoring, disaster management, environmental analysis, and strategic defense reconnaissance."

---

## Slide 5: Research & References
*Visuals: Benchmark metrics (BigEarthNet, RSVQA, CDVQA), literature references.*

**Speaker:** 
"Finally, our approach is deeply grounded in state-of-the-art remote sensing research. 

**[Benchmarks & Fine-Tuning]**
To ensure our models actually understand sensor characteristics and domain terminology, our vision-language components are adapted using the **BigEarthNet** dataset. 

**[Evaluation Standards]**
We built our architecture to natively support and excel at the prescribed evaluation benchmarks. Our single-image captioning and grounding pipelines are benchmarked against **VRSBench** and **RSVQA**. Furthermore, our Bi-Temporal orchestration pipeline is specifically optimized to meet the rigorous standards of the **CDVQA** (Change Detection VQA) framework. 

By unifying these specialized, highly-researched models under one intuitive Agentic Controller, SatQuery AI represents the next generation of conversational Earth Observation. 

Thank you. We are now open for questions and the ISRO/SAC evaluation dataset."
