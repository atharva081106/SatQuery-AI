# SatQuery AI - ISRO / SAC PS 26167

**Smart India Hackathon (SIH) 2026**
**Problem Statement:** Multimodal Remote Sensing Image Analysis (PS 26167)

This project provides a comprehensive, agentic AI-driven web application tailored for the analysis of optical/multispectral, SAR, and bi-temporal remote sensing imagery.

## 🏆 Fulfillment of ISRO/SAC Evaluation Criteria

Our solution strictly adheres to the core requirements and evaluation criteria provided for PS 26167:

### 1. Input Upload and Compatibility Checking
- ✅ **Implementation:** The system features a robust frontend upload portal coupled with the `check_spatial_compatibility` algorithm in the backend. 
- ✅ **Detail:** It automatically computes geographic bounding box intersections (IoU > 15%), extracts ORB keypoints, applies RANSAC homography, and verifies the exact co-registration of image pairs before initiating dual-image analysis pipelines.

### 2. Remote-Sensing-Adapted Vision-Language Component
- ✅ **Implementation:** The platform bypasses generic LLMs by employing specialized, remote-sensing-adapted vision-language pipelines.
- ✅ **Detail:** It utilizes `SatSegNet` for spatial grounding and bespoke Siamese networks for pixel-level difference analysis, strictly avoiding the hallucinations characteristic of generic off-the-shelf multimodal LLMs.

### 3. Specialist Tools for VQA, Grounding, Change Understanding, and Optical-SAR Analysis
- ✅ **Implementation:** Specialized deep learning agents exist for each sub-domain.
- ✅ **Detail:** 
  - **Single-Image VQA:** Answer queries with pinpoint object localizations.
  - **Semantic Grounding:** Bounding box vector extractions (GeoJSON compatible).
  - **Change Understanding:** Siamese bi-temporal difference mapping.
  - **Optical-SAR Analysis:** Cross-modal structural correlation matching to penetrate cloud cover.

### 4. Agentic Controller for Task Routing and Tool Execution
- ✅ **Implementation:** An LLM-powered orchestration engine (`agent_controller.py`).
- ✅ **Detail:** The orchestrator autonomously parses natural language commands, assigns execution to the correct specialist computer-vision model (avoiding monolithic AI failures), aggregates outputs, and dynamically sequences complex geospatial requests.

### 5. Visual Evidence, Confidence Information, and Downloadable Reports
- ✅ **Implementation:** Traceable, high-contrast UI tailored for tactical intelligence.
- ✅ **Detail:** The frontend Trace & Evidence panel visually highlights detections, publishes deterministic model confidence scores, summarizes the execution chain, and allows for one-click exports to official **PDF Audit Reports** for immediate decision-making.

### 6. Implementation Scope & Interactive GUI
- ✅ **Implementation:** Modern, aerospace-themed Next.js Turbopack web application.
- ✅ **Detail:** Handles single optical/multispectral/SAR queries, paired analyses, and bi-temporal change mapping, with a pre-configured suite of 5 Demo Queries to instantly validate system capability.

---

## 👨‍🏫 Mentorship & Evaluation Guidelines

This solution has been engineered keeping the final evaluation datasets in mind. The system is fully equipped to ingest the pre-georeferenced and co-registered Cartosat-2S optical and RISAT SAR image pairs from the ISRO/SAC evaluation subset, compute metrics securely, and provide detailed vector outputs (bounding boxes/masks) for benchmarking.

**ISRO / SAC Project Mentors:**
- **Aminur Hossain:** [aminur@sac.isro.gov.in](mailto:aminur@sac.isro.gov.in)
- **Sanjay K Singh:** [sks@sac.isro.gov.in](mailto:sks@sac.isro.gov.in)
- **S Devakanth Naidu:** [devakanth@sac.isro.gov.in](mailto:devakanth@sac.isro.gov.in)

---

*This document serves as the official integration manifesto mapping the software capability directly to the Smart India Hackathon problem statement.*
