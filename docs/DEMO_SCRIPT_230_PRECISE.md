> [!WARNING]
> **SUPERSEDED HISTORICAL DOCUMENT**
> This script is retained for historical tracking.

# SatQuery AI — Precise 2:30 Demo Script [SUPERSEDED]
**ISRO PS 26167 | SIH 2026 Live Walkthrough**
*Every timestamp is a hard cue. Rehearse with a stopwatch.*

---

## 🎬 [0:00 – 0:13] Landing Page

**Action:** Open homepage — let the hero section fully load and animate in.

**Speaker:**
> "Meet SatQuery AI — a natural-language intelligence platform built for ISRO Problem Statement 26167.
> Ask a satellite anything, in plain English, and get a verified, visual answer."

> **Scroll subtly** past the feature cards. Don't linger — just show depth.

---

## 🔎 [0:13 – 0:21] Query Page Preview

**Action:** Click **"Try Demo Query"** or navigate to `/query`. Let the page load. Don't submit anything yet.

**Speaker:**
> "This is the Query Studio — where natural language becomes satellite intelligence."

> Just show the layout: upload zone, prompt box, trace panel. Move on quickly.

---

## 🗺️ [0:21 – 0:28] Satellite Capture Page

**Action:** Navigate to the **Satellite Map Explorer** (`/acquire` or click "Satellite Map").

**Speaker:**
> "Before querying, we need data. Our built-in Map Explorer lets you capture any region on Earth in seconds."

---

## 🛰️ [0:28 – 0:36] Acquire Button (Pre-login)

**Action:** Zoom into a coastal/port region. Click **"Acquire"** — the auth wall appears.

**Speaker:**
> "Clicking Acquire triggers our secure acquisition flow — authentication required."

---

## 🔐 [0:36 – 0:41] Auth Page — Sign In as SIH Evaluator

**Action:** Sign in using evaluator credentials (e.g., `sih_evaluator` / `SIH2026`). Login completes.

**Speaker:**
> "Signing in as the SIH evaluator. Role-based access grants full analyst privileges."

---

## 📡 [0:41] Acquire — Image Captured → Query Page

**Action:** After login, click **"Acquire"** again. Image captures. System auto-routes to the Query page with the acquired tile preloaded in the upload zone.

**Speaker:**
> "One click — and our live geospatial tile is instantly loaded into the Query Studio. No manual download."

---

## 🌿 [0:41 – 1:02] Land Cover Query — Type & Execute

**Action:** In the prompt box, type:
```
Describe the land cover and detect built-up structures in this scene
```
Hit **Execute**. Watch the Trace Panel animate through routing steps.

**Speaker:**
> "I ask the AI to describe what it sees. Watch the Trace Panel — our Agentic Orchestrator routes this to the SatSegNet semantic segmentation model.
> It identifies six land-cover classes: built-up, vegetation, water, bare soil, roads, and cloud cover.
> Each pixel gets a label — not a guess — with a deterministic confidence score."

> **Point to the output:** segmentation mask overlay + land-cover distribution table.

---

## 📋 [1:02 – 1:10] Demo Queries Panel

**Action:** Open the **"Sample Missions"** / Demo Queries panel from the top nav.

**Speaker:**
> "For our next scenarios, we use our pre-loaded mission presets — each built around a real ISRO operational use case."

> Briefly show the list of 5 missions. Move on.

---

## 🌊 [1:10 – 1:24] Uttarakhand Flash Flood — Bi-Temporal Change Detection

**Action:** Select **"Uttarakhand Flash Flood Catastrophe (Chamoli District)"** preset → click **"Execute Mission"**.

**Speaker:**
> "Use Case 2: Disaster damage assessment. We load a before-and-after image pair from Rishi Ganga Valley — Cartosat-2S optical vs. post-flood Sentinel-2.
> Notice the **Input Compatibility Badge** — before the AI touches the images, it mathematically verifies both tiles cover the exact same coordinates using ORB keypoints and RANSAC homography. Match score passes at 0.87."

> **Pause on badge: `COMPATIBILITY: PASSED`.**

> "Now the Change Detection specialist runs."

---

## 📊 [1:24 – 1:30] Change Detection Output

**Action:** Output renders — show the interactive before/after swipe slider + red bounding boxes + change table.

**Speaker:**
> "Output: newly flooded river channels highlighted in red, an exact percentage of surface transformation, and a breakdown table — water delta, built-up loss, vegetation change.
> Plus a live before/after swipe comparator and a GeoJSON export for QGIS or ISRO Bhuvan."

---

## ☁️ [1:30 – 1:47] Bay of Bengal Monsoon — Optical-SAR Cross-Modal Fusion

**Action:** Select **"Bay of Bengal Monsoon Cloud Penetration"** preset → click **"Execute Mission"**.

**Speaker:**
> "Use Case 3: optical sensors are useless during monsoon — complete cloud cover.
> Here we fuse a cloud-obscured Cartosat optical image with a RISAT-1 C-Band SAR radar tile.
> Radar at 5.4 GHz pierces through clouds. Our orchestrator detects the cross-modal requirement — keyword triggers: 'cloud', 'SAR', 'penetrate' — and routes to the CrossModalAnalysis engine."

---

## 🖼️ [1:47 – 1:54] SAR Fusion Output + Audit PDF Download

**Action:** Output renders — fused canvas visible. Immediately click **"AUDIT PDF"** button.

**Speaker:**
> "Output: hidden maritime vessels and coastline revealed through 100% cloud cover.
> And now — one click. Export Audit PDF."

---

## 📄 [1:54 – 2:02] PDF Preview

**Action:** PDF opens / previews on screen. Scroll slowly through it.

**Speaker:**
> "The Audit PDF contains: the original query, the exact models invoked, confidence scores, spatial metrics, and visual evidence — a complete, shareable intelligence brief.
> Ready to hand to any decision-maker, no technical background required."

---

## 🏠 [2:02 – 2:06] Dashboard

**Action:** Navigate to **Dashboard** (`/dashboard`).

**Speaker:**
> "Everything is tracked in the Dashboard — query history, analysis logs, model usage."

---

## 📈 [2:06 – 2:20] Benchmark Section

**Action:** Scroll to or click the **Benchmarks** section on the dashboard.

**Speaker:**
> "Our models aren't unverified — they are tested against three prescribed public benchmarks:
> **BigEarthNet** for land-cover classification accuracy, **RSVQA** for visual question answering, and **CDVQA** for change detection.
> These numbers are mathematically grounded — not curated for the demo."

> Point to each metric card briefly.

---

## ❓ [2:20 – 2:28] FAQ Section

**Action:** Scroll to the **FAQ** section on the dashboard.

**Speaker:**
> "For evaluators — every technical question is addressed here: how hallucination is prevented, how spatial verification works, and how the agentic routing decisions are made.
> Full transparency."

---

## 🔄 [2:28 – 2:30] Back to Dashboard

**Action:** Scroll back up to the top of the Dashboard. Freeze on the overview.

**Speaker:**
> "SatQuery AI — space-grade intelligence, in plain English. Thank you."

---

## 📋 Hard Cue Sheet (Stopwatch Reference)

| Timestamp | Scene | Action |
|-----------|-------|--------|
| `0:00` | Landing Page | Hero animates in — scroll past features |
| `0:13` | Query Page | Navigate to `/query` — show layout only |
| `0:21` | Satellite Map | Navigate to acquire/map page |
| `0:28` | Acquire (pre-login) | Click Acquire → auth wall appears |
| `0:36` | Auth / Sign In | Login as `sih_evaluator` |
| `0:41` | Acquire + Route | Click Acquire → image → auto-load to query page |
| `0:41–1:02` | Land Cover Query | Type query → execute → show output |
| `1:02` | Demo Queries | Open Sample Missions panel |
| `1:10` | Uttarakhand | Load Flood preset → execute |
| `1:24` | Flood Output | Swipe slider + change table + GeoJSON |
| `1:30` | Bay of Bengal | Load SAR preset → execute |
| `1:47` | SAR Output + PDF | Show fused output → click Audit PDF |
| `1:54` | PDF Preview | Scroll through PDF |
| `2:02` | Dashboard | Navigate to dashboard |
| `2:06` | Benchmarks | Show BigEarthNet / RSVQA / CDVQA metrics |
| `2:20` | FAQ | Show FAQ section |
| `2:28` | Back to Dashboard | Scroll to top — freeze |
| `2:30` | 🎬 CUT | "Thank you." |

---

## ⚠️ Pre-Demo Checklist

- [ ] **Cache AI responses** for land cover query, flood preset, SAR preset — zero live inference wait
- [ ] **Pre-generate Audit PDF** — click should open instantly
- [ ] **Evaluator account** `sih_evaluator` credentials tested and saved in browser
- [ ] **Acquired tile** preloaded or mock-cached — map load can be slow on conference WiFi
- [ ] **Browser zoom** set to 90% — more UI fits on screen
- [ ] **Trace Panel** visible without scrolling — resize browser if needed
- [ ] **Benchmarks section** loaded — confirm numbers match `MODEL_AUDIT_REPORT.md`
