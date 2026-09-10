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

    # 4. Sambhar Salt Lake Desiccation & Wetland Survey (Ramsar Wetland Site, Rajasthan)
    sambhar_img = np.zeros((400, 400, 3), dtype=np.uint8)
    sambhar_img[:] = [190, 185, 175] # Arid lakebed / saline soil
    # Salt crust boundary
    pts_lake = np.array([[50, 80], [120, 50], [290, 70], [360, 160], [340, 290], [220, 360], [100, 330], [40, 220]], np.int32)
    cv2.fillPoly(sambhar_img, [pts_lake], (230, 235, 240)) # Saline crust
    # Brine reservoir / residual hypersaline water body
    pts_brine = np.array([[120, 140], [260, 130], [310, 210], [240, 280], [140, 270]], np.int32)
    cv2.fillPoly(sambhar_img, [pts_brine], (180, 130, 70)) # High salinity shallow water
    # Salt evaporation pans (man-made grid on east shore)
    for px in range(250, 350, 20):
        for py in range(250, 350, 20):
            cv2.rectangle(sambhar_img, (px, py), (px + 16, py + 16), (220, 215, 205), -1)
            cv2.rectangle(sambhar_img, (px, py), (px + 16, py + 16), (130, 110, 90), 1)

    # 5. Bengaluru Urban Expansion (Cartosat-3 Optical - Whitefield IT Corridor)
    bengaluru_img = np.zeros((400, 400, 3), dtype=np.uint8)
    bengaluru_img[:] = [50, 110, 60] # Natural green cover / tree canopy
    # Arterial highway
    cv2.line(bengaluru_img, (0, 200), (400, 200), (140, 140, 145), 8) # Major 6-lane road
    cv2.line(bengaluru_img, (180, 0), (220, 400), (130, 130, 135), 6) # Cross arterial
    # Tech parks / commercial high-rise footprint
    for tx, ty, tw, th in [(40, 40, 70, 60), (250, 50, 90, 80), (60, 240, 80, 70), (260, 250, 95, 85)]:
        cv2.rectangle(bengaluru_img, (tx, ty), (tx + tw, ty + th), (180, 185, 195), -1) # Concrete roof
        cv2.rectangle(bengaluru_img, (tx, ty), (tx + tw, ty + th), (80, 85, 95), 2)
    # Construction earthwork parcel
    cv2.rectangle(bengaluru_img, (140, 60), (210, 130), (70, 130, 180), -1)

    # 6. Navi Mumbai International Airport (NMIA) Construction Corridor (Bi-temporal Pair)
    nmia_t1 = np.zeros((400, 400, 3), dtype=np.uint8)
    nmia_t1[:] = [70, 130, 80] # Agricultural/wetland green baseline
    # Creek interface
    pts_creek = np.array([[0, 280], [140, 290], [280, 330], [400, 320], [400, 400], [0, 400]], np.int32)
    cv2.fillPoly(nmia_t1, [pts_creek], (160, 110, 30))
    # Rural road
    cv2.line(nmia_t1, (0, 120), (400, 140), (160, 160, 165), 3)

    # NMIA T2: Major airport earthwork grading, runway construction, and tarmac infrastructure
    nmia_t2 = nmia_t1.copy()
    # Extensive leveled earthwork & runway strip
    cv2.rectangle(nmia_t2, (30, 80), (370, 180), (180, 185, 195), -1) # Runway base
    cv2.line(nmia_t2, (50, 130), (350, 130), (230, 230, 235), 8) # Active runway centerline
    cv2.rectangle(nmia_t2, (120, 190), (280, 260), (170, 175, 185), -1) # Terminal apron
    # Retain creek
    cv2.fillPoly(nmia_t2, [pts_creek], (160, 110, 30))

    return [
        {
            "id": "uttarakhand_flood",
            "title": "CHAMOLI GLACIER BURST & RISHI GANGA FLOOD",
            "tag": "BI-TEMPORAL CHANGE DETECTION",
            "location": "Rishi Ganga Valley, Uttarakhand (30.4150° N, 79.7340° E)",
            "sensors": "Cartosat-2S PAN (0.65m) / Sentinel-2 MSI (10m L2A)",
            "query": "What changed between these two dates, and where did the change occur?",
            "description": "Bi-temporal analysis capturing catastrophic river channel swelling, debris deposition, and bridge infrastructure washouts.",
            "images": [
                {
                    "name": "chamoli_pre_event_t1.png",
                    "base64": f"data:image/png;base64,{_encode_b64(t1_flood)}",
                    "label": "T1: BASELINE PRE-EVENT"
                },
                {
                    "name": "chamoli_post_event_t2.png",
                    "base64": f"data:image/png;base64,{_encode_b64(t2_flood)}",
                    "label": "T2: POST-DISASTER INUNDATION"
                }
            ]
        },
        {
            "id": "mumbai_port_recon",
            "title": "JNPT & MUMBAI HARBOR STRATEGIC RECON",
            "tag": "SPATIAL GROUNDING & WATER DELINEATION",
            "location": "Jawaharlal Nehru Port, Navi Mumbai (18.9490° N, 72.9510° E)",
            "sensors": "Cartosat-2S High-Resolution 4-Band VNIR (0.65m GSD)",
            "query": "Highlight the water body referred to in the query and locate maritime berths and vessels",
            "description": "Sub-meter optical reconnaissance isolating maritime shipping berths, liquid storage tank farms, and container vessels.",
            "images": [
                {
                    "name": "mumbai_cartosat2s_optical.png",
                    "base64": f"data:image/png;base64,{_encode_b64(mumbai_opt)}",
                    "label": "CARTOSAT-2S PANCHROMATIC (0.65m)"
                }
            ]
        },
        {
            "id": "bay_of_bengal_sar",
            "title": "BAY OF BENGAL MONSOON CLOUD PENETRATION",
            "tag": "OPTICAL–SAR CROSS-MODAL FUSION",
            "location": "Andaman Sea Maritime Corridor (12.3520° N, 92.7840° E)",
            "sensors": "Cartosat-3 Optical (100% Cloud-Cover) + RISAT-1 / EOS-04 C-Band SAR",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "description": "Fuses 100% cloud-obscured optical imagery with RISAT-1 C-band microwave radar backscatter to detect hidden vessels and coastlines.",
            "images": [
                {
                    "name": "cyclone_cloud_obscured_optical.png",
                    "base64": f"data:image/png;base64,{_encode_b64(cloud_opt)}",
                    "label": "OPTICAL (100% MONSOON CLOUD COVER)"
                },
                {
                    "name": "risat1_cband_radar_sar.png",
                    "base64": f"data:image/png;base64,{_encode_b64(sar_radar)}",
                    "label": "RISAT-1 C-BAND SAR (5.4 GHz RADAR)"
                }
            ]
        },
        {
            "id": "bengaluru_urban_sprawl",
            "title": "WHITEFIELD TECH CORRIDOR DIVERSIFICATION",
            "tag": "MULTITASK SCENE & OBJECT LOCALIZATION",
            "location": "Whitefield IT Corridor, Bengaluru (12.9698° N, 77.7499° E)",
            "sensors": "Cartosat-3 Ultra High-Resolution Optical (0.28m GSD)",
            "query": "Describe the land-cover and major objects visible in this image.",
            "description": "Sub-30cm high-density urban analysis isolating commercial complexes, arterial highways, and vegetative buffers with full metrics.",
            "images": [
                {
                    "name": "bengaluru_cartosat3.png",
                    "base64": f"data:image/png;base64,{_encode_b64(bengaluru_img)}",
                    "label": "CARTOSAT-3 SUB-30cm (0.28m GSD)"
                }
            ]
        },
        {
            "id": "navimumbai_airport_trend",
            "title": "NAVI MUMBAI AIRPORT (NMIA) URBAN GROWTH",
            "tag": "BI-TEMPORAL TREND QUANTIFICATION",
            "location": "Ulwe / Panvel Creek, Navi Mumbai (18.9902° N, 73.0684° E)",
            "sensors": "Cartosat-2S / Sentinel-2 Multi-Year Bi-Temporal Pair",
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
            "description": "Multi-year monitoring tracking agricultural and marshland transition into runway grading and international airport terminal footprint.",
            "images": [
                {
                    "name": "nmia_baseline_t1.png",
                    "base64": f"data:image/png;base64,{_encode_b64(nmia_t1)}",
                    "label": "T1: BASELINE MARSHLAND"
                },
                {
                    "name": "nmia_post_runway_t2.png",
                    "base64": f"data:image/png;base64,{_encode_b64(nmia_t2)}",
                    "label": "T2: ACTIVE RUNWAY & TERMINAL"
                }
            ]
        },
        {
            "id": "sambhar_salt_lake",
            "title": "SAMBHAR HYPERSALINE LAKE WETLAND SURVEY",
            "tag": "WETLAND DELINEATION & RFC 7946",
            "location": "Sambhar Lake Ramsar Wetland #464, Rajasthan (26.9010° N, 75.0020° E)",
            "sensors": "Resourcesat-2A LISS-4 Multispectral (5.8m GSD)",
            "query": "Detect water body boundary and calculate total wetland surface area in km²",
            "description": "Delineation of hypersaline lake perimeter, industrial evaporation salt pans, and arid basin boundaries with RFC 7946 GeoJSON export.",
            "images": [
                {
                    "name": "sambhar_lake_liss4.png",
                    "base64": f"data:image/png;base64,{_encode_b64(sambhar_img)}",
                    "label": "RESOURCESAT-2A LISS-4 (5.8m)"
                }
            ]
        }
    ]
