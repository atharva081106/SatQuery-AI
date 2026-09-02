import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_sih_architecture(output_path="sih_technical_approach_diagram.png"):
    # Create 16:9 widescreen canvas at high resolution (300 DPI)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Background
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Colors palette (Crisp SIH Blueprint Palette)
    C_HEADER = "#0A2540"
    C_CONTAINER_BG = "#F8FAFC"
    C_CONTAINER_BORDER = "#94A3B8"
    C_BOX_INPUT = "#EFF6FF"
    C_BOX_INPUT_BORDER = "#3B82F6"
    C_BOX_PROC = "#F1F5F9"
    C_BOX_PROC_BORDER = "#64748B"
    C_BOX_AI = "#F0FDF4"
    C_BOX_AI_BORDER = "#10B981"
    C_BOX_DEC = "#FEF2F2"
    C_BOX_DEC_BORDER = "#EF4444"
    C_BOX_OUT = "#FAF5FF"
    C_BOX_OUT_BORDER = "#8B5CF6"
    C_ARROW = "#0284C7"
    C_TEXT = "#0F172A"
    C_SUBTEXT = "#475569"

    # Title & Subtitle Banner
    ax.text(8.0, 8.55, "SatQuery AI: Autonomous Multimodal Remote Sensing Architecture",
            fontsize=20, fontweight="bold", color=C_HEADER, ha="center", va="center", family="sans-serif")
    ax.text(8.0, 8.22, "SIH 2026 Problem Statement 26167 | ISRO / Space Applications Centre (SAC) | End-to-End Technical Approach",
            fontsize=10.5, fontweight="semibold", color="#64748B", ha="center", va="center", family="sans-serif")

    def draw_container(x, y, w, h, title, subtitle=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.18",
                                      facecolor=C_CONTAINER_BG, edgecolor=C_CONTAINER_BORDER,
                                      linestyle="--", linewidth=1.5, zorder=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.28, title, fontsize=11, fontweight="bold",
                color=C_HEADER, ha="center", va="center", family="sans-serif", zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.48, subtitle, fontsize=8, color="#64748B",
                    ha="center", va="center", family="sans-serif", zorder=3)

    def draw_module(x, y, w, h, title, desc, facecolor, edgecolor, tag=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=facecolor, edgecolor=edgecolor,
                                      linewidth=1.4, zorder=2)
        ax.add_patch(rect)
        if tag:
            tag_box = patches.FancyBboxPatch((x + w - 0.85, y + h - 0.22), 0.8, 0.18,
                                             boxstyle="round,pad=0.02,rounding_size=0.04",
                                             facecolor=edgecolor, edgecolor="none", zorder=3)
            ax.add_patch(tag_box)
            ax.text(x + w - 0.45, y + h - 0.13, tag, fontsize=6.5, fontweight="bold",
                    color="#FFFFFF", ha="center", va="center", family="sans-serif", zorder=4)

        ax.text(x + 0.15, y + h - 0.28, title, fontsize=9.2, fontweight="bold",
                color=C_TEXT, ha="left", va="center", family="sans-serif", zorder=3)
        ax.text(x + 0.15, y + (h - 0.28)/2, desc, fontsize=7.2, color=C_SUBTEXT,
                ha="left", va="center", family="sans-serif", zorder=3, linespacing=1.25)

    def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style="->", rad=0.0, lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=4, shrinkB=4,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=4)

    # 1. INPUTS & DATA SOURCES (X: 0.5 -> 3.2)
    draw_container(0.5, 0.6, 2.75, 7.3, "1. INPUTS & SENSORS", "Multimodal Observation Scope")
    draw_module(0.65, 5.85, 2.45, 1.45, "Optical / Multispectral",
                "• Sentinel-2 (10m-60m)\n• Cartosat-2S Panchromatic (0.6m)\n• Multi-band GeoTIFF / TIFF / PNG",
                C_BOX_INPUT, C_BOX_INPUT_BORDER, "OPTICAL")
    draw_module(0.65, 4.10, 2.45, 1.50, "Synthetic Aperture Radar",
                "• RISAT-1 C-Band SAR\n• Sentinel-1 C-Band Microwave\n• Day/Night Cloud Penetration",
                C_BOX_INPUT, C_BOX_INPUT_BORDER, "SAR RADAR")
    draw_module(0.65, 2.35, 2.45, 1.50, "Multitemporal Pairs",
                "• T1 Baseline & T2 Post-Disaster\n• Co-registered Geospatial Footprints\n• Flood & Landslide Disasters",
                C_BOX_INPUT, C_BOX_INPUT_BORDER, "BI-TEMPORAL")
    draw_module(0.65, 0.85, 2.45, 1.25, "Natural Language Query",
                "• Free-text operational questions\n• Contextual multi-turn followups\n• GIS feature targeting commands",
                C_BOX_INPUT, C_BOX_INPUT_BORDER, "TEXT QUERY")

    # 2. DATA PROCESSING & SPATIAL GUARD (X: 3.55 -> 6.15)
    draw_container(3.55, 0.6, 2.60, 7.3, "2. DATA PROCESSING", "Geospatial Alignment & Guard")
    draw_module(3.70, 5.45, 2.30, 1.80, "Rasterio GDAL Core",
                "• CRS Resolution (EPSG:4326)\n• Affine Transform & Bounds\n• 16-bit to 8-bit Normalization\n• Bounding Box IoU Overlap",
                C_BOX_PROC, C_BOX_PROC_BORDER, "METADATA")
    draw_module(3.70, 2.75, 2.30, 2.40, "Spatial Compatibility Guard",
                "• ORB Feature Extraction (1200 pts)\n• Lowe's Ratio Matching (0.78)\n• RANSAC Homography Matrix H\n• Inlier Verification (Threshold >= 18)\n• Zero-Hallucination Prevention",
                "#FEF2F2", "#DC2626", "USP 1: GUARD")
    draw_module(3.70, 0.85, 2.30, 1.60, "Streaming Decimator",
                "• Windowed raster chunking\n• Resolves GPU OOM Bottlenecks\n• Multi-resolution Pyramid Cache\n• Sub-100ms Ingestion Latency",
                C_BOX_PROC, C_BOX_PROC_BORDER, "STREAMING")

    # 3. AI/ML ENGINE & MODEL REGISTRY (X: 6.45 -> 9.65)
    draw_container(6.45, 0.6, 3.20, 7.3, "3. AI/ML SPECIALIST REGISTRY", "Agentic Controller & Adapted Models")
    draw_module(6.60, 5.80, 2.90, 1.45, "Agentic Controller Router",
                "• Task Planning & Query Classification\n• Dynamic Specialist Model Sequencing\n• Input Scope Compliance Enforcement\n• Auditable Trace Metadata Generation",
                "#EEF2FF", "#4F46E5", "ORCHESTRATOR")
    draw_module(6.60, 4.45, 2.90, 1.25, "Adapted BLIP-VQA Engine",
                "• ViT-B/16 Vision Transformer + BERT\n• Fine-Tuned on BigEarthNet & RSVQA\n• Spectral Feature Attention Weights",
                C_BOX_AI, C_BOX_AI_BORDER, "FINE-TUNED")
    draw_module(6.60, 3.10, 2.90, 1.25, "Bi-Temporal Change Engine",
                "• Cross-Attention Differential Siamese\n• Pixel Inundation Difference Vectors\n• Morphological Flood/Damage Contours",
                C_BOX_AI, C_BOX_AI_BORDER, "CHANGE VQA")
    draw_module(6.60, 1.75, 2.90, 1.25, "Cross-Modal Optical-SAR",
                "• Structural Gradient Correlator\n• Microwave Backscatter Penetration\n• Day/Night All-Weather Grounding",
                C_BOX_AI, C_BOX_AI_BORDER, "SAR FUSION")
    draw_module(6.60, 0.80, 2.90, 0.85, "Spatial Grounding & Captioning",
                "• Text-guided contour bounding\n• Holistic remote-sensing scene description",
                C_BOX_AI, C_BOX_AI_BORDER, "GROUNDING")

    # 4. DECISION ENGINE & EVIDENCE SYNTHESIS (X: 9.95 -> 12.55)
    draw_container(9.95, 0.6, 2.60, 7.3, "4. DECISION & EVIDENCE", "Evidence Grounding & GIS Export")
    draw_module(10.10, 5.65, 2.30, 1.60, "Disparity Gate & Confidence",
                "• Spatial Coherence Scoring (0.0-1.0)\n• Softmax Probability Calibration\n• Disparity Alert Card Emitter\n• Guaranteed Rejection on Mismatch",
                C_BOX_DEC, C_BOX_DEC_BORDER, "SAFETY GATE")
    draw_module(10.10, 3.45, 2.30, 1.95, "XAI Heatmap Generator",
                "• Grad-CAM Activation Mapping\n• Precise Pixel Clusters Driving Predictions\n• Color-Coded Spectral Attention\n• Trace Telemetry Integration",
                C_BOX_DEC, C_BOX_DEC_BORDER, "VISUAL XAI")
    draw_module(10.10, 0.80, 2.30, 2.40, "Tactical GeoJSON Engine",
                "• OpenCV Morphological Vectorization\n• WGS84 Geographic Projection\n• RFC 7946 Standard Compliance\n• Direct Ingestion for QGIS & Bhuvan\n• Instant 1-Click Vector Export",
                "#FAF5FF", "#7C3AED", "USP 3: GEOJSON")

    # 5. ACTIONS, HARDWARE & TACTICAL DASHBOARD (X: 12.85 -> 15.5)
    draw_container(12.85, 0.6, 2.65, 7.3, "5. ACTIONS & DASHBOARD", "Hardware & Tactical Execution")
    draw_module(13.00, 5.65, 2.35, 1.60, "Air-Gapped Sovereign Node",
                "• Local Edge Tactical Unit / Laptop\n• Consumer GPU Supported (<6GB VRAM)\n• CPU-Only Fallback Engine\n• 100% Offline (Zero Cloud Leakage)",
                C_BOX_OUT, C_BOX_OUT_BORDER, "AIR-GAPPED")
    draw_module(13.00, 3.45, 2.35, 1.95, "Interactive Swipe Slider",
                "• Sub-Pixel Draggable Divider Curtain\n• Instant T1 vs T2 Flood Overlay\n• Optical vs Penetrating SAR Radar\n• Touch & Mouse Responsive",
                "#ECFDF5", "#059669", "USP 2: SWIPE")
    draw_module(13.00, 0.80, 2.35, 2.40, "Next.js 15 HUD & Reports",
                "• Aerospace Dark Telemetry Interface\n• Real-Time Streamlined Execution Trace\n• Downloadable PDF Mission Audits (jsPDF)\n• Direct QGIS / ArcGIS Vector Export",
                C_BOX_OUT, C_BOX_OUT_BORDER, "TACTICAL HUD")

    # Connective Arrows between major pipeline columns
    draw_arrow(3.10, 6.55, 3.70, 6.35) # Optical -> Rasterio
    draw_arrow(3.10, 4.85, 3.70, 3.95) # SAR -> Spatial Guard
    draw_arrow(3.10, 3.10, 3.70, 3.95) # Temporal -> Spatial Guard
    draw_arrow(4.85, 5.45, 4.85, 5.15) # Rasterio -> Spatial Guard
    draw_arrow(6.00, 3.95, 6.60, 6.50) # Spatial Guard -> Agentic Controller
    draw_arrow(3.10, 1.45, 6.60, 6.50, rad=-0.2) # Query -> Agentic Controller

    # Agentic Controller -> Specialist Engines
    draw_arrow(8.05, 5.80, 8.05, 5.70)
    draw_arrow(8.05, 4.45, 8.05, 4.35)
    draw_arrow(8.05, 3.10, 8.05, 3.00)
    draw_arrow(8.05, 1.75, 8.05, 1.65)

    # Engines -> Decision & Synthesis
    draw_arrow(9.50, 5.05, 10.10, 6.45) # BLIP VQA -> Disparity/Confidence
    draw_arrow(9.50, 3.70, 10.10, 4.45) # Change Engine -> Heatmap
    draw_arrow(9.50, 2.35, 10.10, 2.00) # Engines -> GeoJSON

    # Decision -> Dashboard & Hardware
    draw_arrow(12.40, 6.45, 13.00, 6.45) # Disparity Gate -> Air-Gapped Node
    draw_arrow(12.40, 4.45, 13.00, 4.45) # Heatmap -> Swipe Slider
    draw_arrow(12.40, 2.00, 13.00, 2.00) # GeoJSON -> Next.js HUD / GIS Export

    # Safety Control Loop Feedback (disparity alert card)
    draw_arrow(10.10, 6.00, 6.00, 4.50, color="#DC2626", rad=-0.35, style="<-", lw=1.5)
    ax.text(8.05, 0.35, "⚠️ Safety Feedback Loop: Non-Co-Registered Image Pairs Aborted & Alert Card Dispatched",
            fontsize=8.5, fontweight="bold", color="#DC2626", ha="center", va="center", family="sans-serif")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Diagram successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    draw_sih_architecture()
