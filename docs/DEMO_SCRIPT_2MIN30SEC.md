> [!WARNING]
> **SUPERSEDED HISTORICAL DOCUMENT**
> This script is retained for historical tracking.

# SatQuery AI — 2:30 Demonstration Script [SUPERSEDED]
**ISRO PS 26167 | Live Prototype Walkthrough**

> **Pacing guide:** ~130 words per minute. Total word count ≈ 320 words.
> Each timestamp block is a hard cut-off — rehearse with a stopwatch.

---

## 🎬 [0:00 – 0:20] Opening Hook

**Action:** Show the SatQuery AI homepage on screen.

**Speaker:**
> "Imagine asking a satellite *'Show me the ships hiding under monsoon clouds in the Bay of Bengal'* — in plain English — and getting a verified, visual answer in seconds.
> That's SatQuery AI. Let me show you how it works."

---

## 🛰️ [0:20 – 0:50] Step 1 — Acquire Live Data (Map Explorer)

**Action:** Click **"Satellite Map"** → zoom into a port or coastal region → capture a tile.

**Speaker:**
> "First, data. Instead of hunting through external portals, our built-in Satellite Map Explorer lets you navigate any region on the globe and capture a live geospatial tile instantly.
> One click — and this image is now ready for AI analysis. No manual download, no pre-processing."

---

## 🔍 [0:50 – 1:20] Step 2 — Object Detection Query (VQA + Grounding)

**Action:** Pass captured image to the Query page → type: *"Highlight the maritime vessels and storage tanks."*

**Speaker:**
> "Now I ask the AI a plain-English question. Watch the **Trace Panel** on the right — our Agentic Orchestrator understands the request and routes the image to our Object Grounding specialist.
> Instead of a text answer, it draws **exact bounding boxes** around the ships and oil silos — with deterministic confidence scores. The AI proves its work, visually."

---

## 🌊 [1:20 – 1:55] Step 3 — Change Detection (Flood Damage)

**Action:** Launch the **"Uttarakhand Flash Flood"** demo preset — two images load side by side.

**Speaker:**
> "What if we need to assess disaster damage? We load a before-and-after image pair.
> Before the AI even starts, notice this **Input Compatibility Badge** — it mathematically verifies both images cover the exact same location using ORB keypoints and RANSAC homography.
> Once confirmed, our Change Detection specialist highlights the *newly flooded river channels* in red. Precise. Verified. Instant."

---

## ☁️ [1:55 – 2:15] Step 4 — Cloud Penetration (SAR Fusion)

**Action:** Launch the **"Bay of Bengal Monsoon"** demo preset — optical + SAR images load.

**Speaker:**
> "Standard cameras are blind in monsoon season. Here we combine a cloud-covered optical image with a SAR radar image. Radar sees through clouds.
> SatQuery AI fuses both modalities, cuts through the cloud cover, and detects the hidden vessels underneath — impossible for any single-sensor tool."

---

## 📄 [2:15 – 2:30] Step 5 — Export Audit PDF

**Action:** Click the **"AUDIT PDF"** button → PDF preview appears.

**Speaker:**
> "Finally — intelligence is only useful if it can be shared. One click generates an official Audit PDF: query, models used, confidence scores, and visual evidence — ready to hand to a decision-maker.
> From satellite to actionable report, in under 60 seconds. Thank you."

---

## 📋 Speaker Notes & Pacing Table

| Timestamp | Key Action | Risk if Slow |
|-----------|-----------|--------------|
| 0:00–0:20 | Hook — no clicks | Losing attention |
| 0:20–0:50 | Map capture | Slow map load — use cached tile |
| 0:50–1:20 | Type query + show trace | AI latency — pre-run & cache result |
| 1:20–1:55 | Flood preset | Click preset, don't explain UI |
| 1:55–2:15 | SAR preset | Skip SAR theory — show output only |
| 2:15–2:30 | PDF button | Pre-generate PDF to avoid delay |

**Rehearsal tip:** Record yourself on your phone. If any block runs over by 5 seconds, tighten the spoken line — not the demo action.

**IMPORTANT:** Pre-load all demo presets and cache AI responses before the actual demonstration to avoid live inference delays.
