"""
Pre-packaged sample missions for SatQuery AI live demonstration.
Generates realistic tactical remote sensing image pairs and metadata
with rich co-registered structural features for instant 1-click execution.
"""
import base64
import numpy as np
import cv2

def _encode_b64(cv_img):
    _, buffer = cv2.imencode('.png', cv_img)
    return base64.b64encode(buffer).decode('utf-8')

def generate_sample_missions():
    # 1. Uttarakhand Flash Flood Scenario (Bi-temporal Pair)
    # Realistic co-registered landscape with roads, infrastructure, and terrain parcel grid
    base_terrain = np.zeros((400, 400, 3), dtype=np.uint8)
    base_terrain[:] = [45, 90, 55] # Mountain valley terrain

    # Structured terrain parcels and agricultural valley terraces
    for i in range(20, 380, 40):
        for j in range(20, 380, 40):
            c = (i * 9 + j * 3) % 120 + 60
            cv2.rectangle(base_terrain, (i, j), (i + 34, j + 34), (c - 15, c + 15, c - 20), -1)
            cv2.rectangle(base_terrain, (i, j), (i + 34, j + 34), (30, 45, 30), 1)

    # Mountain highway and river valley corridor
    cv2.line(base_terrain, (0, 160), (400, 240), (190, 190, 195), 4) # Highway
    cv2.line(base_terrain, (180, 0), (220, 400), (180, 180, 185), 3) # Valley Road

    # Bridges & settlement buildings
    for bx, by in [(120, 80), (280, 100), (80, 260), (320, 280), (190, 170)]:
        cv2.rectangle(base_terrain, (bx, by), (bx + 22, by + 18), (170, 175, 185), -1)
        cv2.rectangle(base_terrain, (bx, by), (bx + 22, by + 18), (40, 40, 45), 1)

    # T1 Pre-flood baseline: clear river
    t1_flood = base_terrain.copy()
    pts_river = np.array([[30, 0], [90, 100], [180, 180], [240, 260], [310, 400]], np.int32)
    cv2.polylines(t1_flood, [pts_river], False, (175, 120, 40), 10)

    # T2 Post-flood event: massive swollen river mud corridor and washed out structures
    t2_flood = base_terrain.copy()
    cv2.polylines(t2_flood, [pts_river], False, (60, 110, 170), 55) # Swollen mud inundation
    cv2.ellipse(t2_flood, (180, 180), (75, 45), 25, 0, 360, (50, 95, 150), -1) # Flood deposit
    cv2.circle(t2_flood, (190, 170), 25, (55, 100, 160), -1) # Submerged settlement

    # 2. Mumbai Port Recon (Cartosat-2S High-Res Optical)
    mumbai_opt = np.zeros((400, 400, 3), dtype=np.uint8)
    mumbai_opt[:] = [150, 90, 35] # Coastal sea
    # Port terminal landmass
    cv2.rectangle(mumbai_opt, (160, 0), (400, 400), (95, 100, 105), -1)
    # Shipping berths and piers extending into water
    cv2.rectangle(mumbai_opt, (90, 70), (160, 120), (115, 120, 125), -1)
    cv2.rectangle(mumbai_opt, (80, 230), (160, 280), (115, 120, 125), -1)
    # Moored cargo ships
    cv2.rectangle(mumbai_opt, (35, 75), (90, 115), (200, 205, 215), -1)
    cv2.rectangle(mumbai_opt, (25, 235), (80, 275), (220, 160, 90), -1)
    # Storage tank clusters
    for x in range(210, 370, 42):
        for y in range(50, 360, 55):
            cv2.circle(mumbai_opt, (x, y), 15, (225, 225, 225), -1)
            cv2.circle(mumbai_opt, (x, y), 15, (60, 60, 60), 2)

    # 3. Bay of Bengal (Cloud-Obscured Optical + Penetrating RISAT-1 C-Band SAR)
    # Shared underlying island and coastline features
    island_base = np.zeros((400, 400, 3), dtype=np.uint8)
    island_base[:] = [140, 80, 25] # Ocean
    pts_island = np.array([[130, 90], [270, 110], [330, 250], [240, 340], [110, 270]], np.int32)
    cv2.fillPoly(island_base, [pts_island], (60, 110, 70))
    # Harbor installation on island
    cv2.rectangle(island_base, (220, 200), (280, 260), (160, 165, 175), -1)
    for ix in range(140, 260, 25):
        cv2.circle(island_base, (ix, ix + 10), 4, (200, 200, 210), -1)

    # Optical: 80% heavy cloud obscuration
    cloud_opt = island_base.copy()
    for cx, cy, r in [(140, 130, 100), (270, 190, 120), (190, 290, 110)]:
        cv2.circle(cloud_opt, (cx, cy), r, (245, 245, 250), -1)
    cv2.GaussianBlur(cloud_opt, (31, 31), 0, dst=cloud_opt)

    # SAR: Radar returns penetrating cloud layer
    sar_radar = np.zeros((400, 400, 3), dtype=np.uint8)
    noise = np.random.randint(15, 30, (400, 400, 3), dtype=np.uint8)
    sar_radar = cv2.add(sar_radar, noise)
    cv2.fillPoly(sar_radar, [pts_island], (170, 170, 175))
    cv2.rectangle(sar_radar, (220, 200), (280, 260), (230, 230, 240), -1) # High radar reflectivity
    # Corner reflectors (vessels and navigational aids)
    cv2.circle(sar_radar, (80, 110), 6, (255, 255, 255), -1)
    cv2.circle(sar_radar, (340, 190), 7, (255, 255, 255), -1)
    cv2.circle(sar_radar, (95, 320), 6, (255, 255, 255), -1)

    return [
        {
            "id": "uttarakhand_flood",
            "title": "UTTARAKHAND FLASH FLOOD",
            "tag": "BI-TEMPORAL CHANGE DETECTION",
            "location": "Rishi Ganga Valley, Uttarakhand (30.41° N, 79.73° E)",
            "sensors": "Cartosat-2S / Sentinel-2 Bi-Temporal Pair",
            "query": "Run Change Detection between pre-flood baseline and post-flood event",
            "description": "Bi-temporal sequence capturing catastrophic river swelling, debris flow, and structural displacement.",
            "images": [
                {
                    "name": "uttarakhand_pre_event_t1.png",
                    "base64": f"data:image/png;base64,{_encode_b64(t1_flood)}",
                    "label": "T1: BASELINE ACQUISITION"
                },
                {
                    "name": "uttarakhand_post_event_t2.png",
                    "base64": f"data:image/png;base64,{_encode_b64(t2_flood)}",
                    "label": "T2: POST-DISASTER TILE"
                }
            ]
        },
        {
            "id": "mumbai_port_recon",
            "title": "MUMBAI HARBOR & DOCKS RECON",
            "tag": "SINGLE-IMAGE VQA & GROUNDING",
            "location": "Jawaharlal Nehru Port, Navi Mumbai (18.95° N, 72.95° E)",
            "sensors": "Cartosat-2S High-Resolution Optical (0.65m GSD)",
            "query": "Highlight industrial storage facilities, maritime docks, and cargo vessels",
            "description": "Sub-meter optical spatial reconnaissance isolating maritime shipping berths and cylindrical storage infrastructure.",
            "images": [
                {
                    "name": "mumbai_cartosat2s_optical.png",
                    "base64": f"data:image/png;base64,{_encode_b64(mumbai_opt)}",
                    "label": "CARTOSAT-2S PANCHROMATIC"
                }
            ]
        },
        {
            "id": "bay_of_bengal_sar",
            "title": "BAY OF BENGAL MONSOON OBSCURED",
            "tag": "OPTICAL–SAR CROSS-MODAL FUSION",
            "location": "Andaman Sea Corridor (12.35° N, 92.78° E)",
            "sensors": "Cartosat Optical + RISAT-1 C-Band SAR Co-Registered",
            "query": "Penetrate cloud cover using SAR radar backscatter channels and extract obscured maritime features",
            "description": "Cross-modal pair demonstrating 100% cloud penetration via RISAT-1 C-band microwave radar to reveal hidden vessels and coastline.",
            "images": [
                {
                    "name": "cyclone_cloud_obscured_optical.png",
                    "base64": f"data:image/png;base64,{_encode_b64(cloud_opt)}",
                    "label": "OPTICAL (CLOUD OBSCURED)"
                },
                {
                    "name": "risat1_cband_radar_sar.png",
                    "base64": f"data:image/png;base64,{_encode_b64(sar_radar)}",
                    "label": "RISAT-1 SAR (PENETRATED)"
                }
            ]
        }
    ]
