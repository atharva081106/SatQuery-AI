# SatQuery AI - 5-Minute Pitch & Demo Script
**ISRO / SAC PS 26167: Multimodal Remote Sensing Image Analysis**

*This script is balanced for both non-expert audiences (using clear analogies) and technical judges (including precise domain terminology).*

---

## ⏱️ PART 1: The Pitch - Problem, Solution & Architecture (0:00 - 1:15)

**[0:00 - 0:30] The Problem: Why Current AI Fails in Space**
**Speaker:** "Good morning everyone. Today we are addressing ISRO Problem Statement 26167. 
Satellite imagery is incredibly powerful, but using AI to analyze this data is extremely difficult for non-experts. If you ask a generic AI chat model to analyze a satellite image, it often hallucinates—it guesses the answer because it wasn't trained on geospatial physics. Furthermore, specialized tools are fragmented. You need one tool for optical images, a completely different tool for radar, and another for change detection."

**[0:30 - 1:15] The Solution: An 'AI Hospital' of Specialists**
**Speaker:** "Our solution is **SatQuery AI**. Instead of building one generic AI model that tries to do everything and fails, we built an **Agentic Orchestrator**. 
Think of our system like a hospital. The user types their query in plain English—like speaking to a receptionist. Our 'Agentic Controller' understands the request and automatically routes the images to the correct specialist doctor. We have one specialist model for object detection (SatSegNet), another specifically for comparing timelines (Siamese Networks), and another for penetrating clouds using SAR radar."

---

## ⏱️ PART 2: Live Prototype Demonstration (1:15 - 4:30)

**[1:15 - 1:45] The 'Acquire' Workflow: Fetching Live Data**
* **Action:** Click "Satellite Map" on the homepage to open the Map Explorer (Acquire Page).
* **Speaker:** "Before we even query the AI, we need data. Unlike other platforms where you must manually hunt for satellite tiles, SatQuery AI features a built-in Satellite Map Explorer. 
A user can simply navigate the globe, zoom into any tactical region—like a port or an airfield—and instantly capture a live geospatial tile. We can then send this exact captured image directly into the AI for analysis with a single click."

**[1:45 - 2:30] Use Case 1: Finding Objects (Single-Image VQA & Grounding)**
* **Action:** Pass the captured image to the Query page, and type a query like: "Highlight the maritime vessels and storage tanks."
* **Speaker:** "Now we ask the AI to find specific objects in the image we just captured. 
Notice the Trace Panel on the right. Our orchestrator understood the assignment and sent the image to our Object Grounding model. Instead of just giving us a text answer, the AI draws exact bounding boxes around the ships and oil silos, proving its work visually with a deterministic confidence score."

**[2:30 - 3:15] Use Case 2: Spotting Disasters (Bi-Temporal Change Detection)**
* **Action:** Launch the **"Uttarakhand Flash Flood"** demo preset.
* **Speaker:** "What if we want to see the damage from a flood? We upload a 'before' and 'after' image. 
*Pause and point to the UI:* Notice the **Input Compatibility Check** badge. Before the AI even looks at the images, it uses advanced math—specifically ORB keypoints and RANSAC homography—to mathematically prove these two images are of the exact same geographic location. Once verified, it passes the images to our Change Detection specialist, which highlights the exact newly flooded river channels."

**[3:15 - 4:00] Use Case 3: Seeing Through Clouds (Optical-SAR Fusion)**
* **Action:** Launch the **"Bay of Bengal Monsoon Cloud Penetration"** demo preset.
* **Speaker:** "Standard optical satellites are useless during monsoon season because they can't see through clouds. 
Here, we upload a completely clouded image, alongside a SAR (Synthetic Aperture Radar) image. Radar can pierce through clouds. Our system recognizes this cross-modal requirement. It fuses the radar data with the optical data, cutting entirely through the cloud cover to detect hidden maritime vessels underneath."

**[4:00 - 4:30] The Output: Printable Evidence (Audit PDF)**
* **Action:** Click the **"AUDIT PDF"** button.
* **Speaker:** "Intelligence is only useful if it can be shared. For any query, a non-expert user can simply click 'Export Audit PDF'. SatQuery AI instantly generates an official report containing the original query, the exact AI models used, confidence metrics, and visual evidence. It bridges the gap between complex remote-sensing science and immediate, actionable reports."

---

## ⏱️ PART 3: Conclusion (4:30 - 5:00)

**[4:30 - 5:00] Final Wrap-up**
**Speaker:** "In just 5 minutes, we have demonstrated a system that allows users to seamlessly acquire satellite data and translate plain English into advanced tactical analysis. By orchestrating specialized remote-sensing models behind a simple interface, SatQuery AI fulfills every mandate of PS 26167—making space-grade intelligence accessible to everyone. Thank you."
