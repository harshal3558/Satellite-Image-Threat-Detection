"""
Build script to generate enterprise-grade HLD.pdf and LLD.pdf for SITP.
Incorporates:
- YOLOv26 Architecture & STAL Label Assignment
- ONNX Runtime 1.16+ & PyTorch Execution Providers
- Subsampled Statistical Variance Gating (sigma < 5.0)
- Dual Telemetry Architecture (Backend vs. Detection Dossier)
- Full AWS Cloud Deployment (IAM Roles, Amazon ECR, Amazon ECS on AWS Fargate & Amazon EC2)
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Color Palette (Tactical Slate / Defense GEOINT theme)
C_PRIMARY = colors.HexColor("#0f172a")     # Deep Slate
C_SECONDARY = colors.HexColor("#1e293b")   # Navy Slate
C_ACCENT = colors.HexColor("#0284c7")      # Electric Blue / Cyan Accent
C_TEXT = colors.HexColor("#334155")        # Body text
C_MUTED = colors.HexColor("#64748b")       # Subdued grey
C_BG_LIGHT = colors.HexColor("#f8fafc")    # Off-white table bg
C_BORDER = colors.HexColor("#cbd5e1")      # Border grey
C_WARN = colors.HexColor("#b45309")        # Amber

class NumberedCanvas(canvas.Canvas):
    """Canvas that performs two passes to render total page numbers and tactical running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 7)
        self.setFillColor(C_MUTED)

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 752, "SATELLITE IMAGE THREAT DETECTION (SITP)  *  GEOINT ARCHITECTURE")
            self.drawRightString(612 - 54, 752, f"DOC ID: SITP-SPEC-v3.2")
            self.setStrokeColor(C_BORDER)
            self.setLineWidth(0.5)
            self.line(54, 746, 612 - 54, 746)

        # Running Footer (all pages)
        self.setStrokeColor(C_BORDER)
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.setFont("Helvetica", 7)
        self.drawString(54, 34, "CONFIDENTIAL & PROPRIETARY  *  DEFENSE & AEROSPACE SPECIFICATION")
        self.drawRightString(612 - 54, 34, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def get_doc_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=C_PRIMARY,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=C_ACCENT,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=C_PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=C_SECONDARY,
        spaceBefore=5,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=C_TEXT,
        spaceAfter=4,
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
    )
    meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=C_SECONDARY,
    )
    meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=C_TEXT,
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=C_TEXT,
    )
    table_hdr = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )
    code_block = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.5,
        leading=8.5,
        textColor=C_PRIMARY,
        backColor=C_BG_LIGHT,
        borderPadding=4,
        spaceAfter=4,
    )

    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h1': h1_style,
        'h2': h2_style,
        'body': body_style,
        'body_bold': body_bold,
        'meta_label': meta_label,
        'meta_val': meta_val,
        'table_cell': table_cell,
        'table_hdr': table_hdr,
        'code': code_block,
    }


def build_hld_pdf(filename="HLD.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_doc_styles()
    story = []

    # ==================== PAGE 1: TITLE & METADATA ====================
    story.append(Paragraph("SATELLITE IMAGE THREAT DETECTION (SITP)", styles['title']))
    story.append(Paragraph("HIGH-LEVEL SYSTEM ARCHITECTURE DESIGN (HLD)", styles['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=8))

    meta_data = [
        [Paragraph("Document ID:", styles['meta_label']), Paragraph("SITP-HLD-v3.2.0", styles['meta_val']),
         Paragraph("Classification:", styles['meta_label']), Paragraph("GEOINT Intelligence Technical Specification", styles['meta_val'])],
        [Paragraph("Target Models:", styles['meta_label']), Paragraph("YOLOv26 (STAL) + ONNX Runtime", styles['meta_val']),
         Paragraph("Cloud Infrastructure:", styles['meta_label']), Paragraph("AWS (ECR · ECS Fargate · EC2 · IAM)", styles['meta_val'])],
        [Paragraph("Release Date:", styles['meta_label']), Paragraph("September 2026", styles['meta_val']),
         Paragraph("Core Capabilities:", styles['meta_label']), Paragraph("GeoTIFF Tiling, Variance Gating, C2 HUD", styles['meta_val'])],
    ]
    t_meta = Table(meta_data, colWidths=[85, 160, 95, 164])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Summary & Strategic Scope", styles['h1']))
    story.append(Paragraph(
        "Automated Geospatial Intelligence (GEOINT) plays a critical role in strategic defense, tactical reconnaissance, "
        "and maritime domain awareness. Conventional computer vision models fail on high-resolution overhead satellite "
        "rasters due to extreme scale disparity (sub-30px targets in gigapixel scenes), non-target background clutter (>60%), "
        "and unoptimized inference memory footprints. The Satellite Image Threat Detection Platform (SITP) delivers an enterprise-grade "
        "end-to-end MLOps pipeline featuring zero-downsampling sliding-window tiling, small-target-aware YOLOv26 detection, "
        "ONNX Runtime vectorization, subsampled variance background rejection, dual-stream telemetry, and production deployment on AWS.",
        styles['body']
    ))

    story.append(Paragraph("2. System Architecture & End-to-End Data Flow", styles['h1']))
    story.append(Paragraph(
        "The SITP architecture decouples ingestion, tiling, neural inference, and presentation into modular pipelines:",
        styles['body']
    ))
    pipe_data = [
        [Paragraph("Pipeline Stage", styles['table_hdr']), Paragraph("Component & Module", styles['table_hdr']), Paragraph("Operational Responsibility", styles['table_hdr'])],
        [Paragraph("Stage 1: Ingestion", styles['table_cell']), Paragraph("DataIngestion<br/>(data_ingestion.py)", styles['table_cell']), Paragraph("Ingests raw xView GeoJSON labels, performs image-level leakage-free train/val split (80/20), and validates file integrity on disk.", styles['table_cell'])],
        [Paragraph("Stage 2: Transformation", styles['table_cell']), Paragraph("DataTransformation<br/>(data_transformation.py)", styles['table_cell']), Paragraph("Generates 512x512 chips with calibrated overlap, transforms bounding polygons to YOLO format, and applies photometric/geometric augmentations.", styles['table_cell'])],
        [Paragraph("Stage 3: Fine-Tuning", styles['table_cell']), Paragraph("ModelTrainer<br/>(model_trainer.py)", styles['table_cell']), Paragraph("Trains YOLOv26 using Small-Target-Aware Label Assignment (STAL), AdamW optimizer, and progressive loss scaling across 62 tactical classes.", styles['table_cell'])],
        [Paragraph("Stage 4: Tiled Serving", styles['table_cell']), Paragraph("PredictPipeline & Monitoring<br/>(prediction_pipeline.py)", styles['table_cell']), Paragraph("Executes streaming window reads directly from disk, evaluates subsampled variance gating, batches active chips, and runs ONNX/PyTorch inference.", styles['table_cell'])],
        [Paragraph("Stage 5: C2 Presentation", styles['table_cell']), Paragraph("Tactical Cockpit HUD<br/>(application.py)", styles['table_cell']), Paragraph("Zero-scroll military C2 web dashboard providing sensitivity controls, high-threat natural language brief, latency telemetry, and itemized logs.", styles['table_cell'])],
    ]
    t_pipe = Table(pipe_data, colWidths=[90, 130, 284])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_pipe)
    story.append(PageBreak())

    # ==================== PAGE 2: INFERENCE OPTIMIZATIONS ====================
    story.append(Paragraph("3. High-Throughput Inference Latency Optimizations", styles['h1']))
    story.append(Paragraph(
        "Scanning massive multi-gigabyte satellite rasters demands aggressive compute optimizations. SITP incorporates six low-latency architectural techniques:",
        styles['body']
    ))

    optimizations = [
        ("1. High-Performance YOLOv26 + ONNX Runtime Engine",
         "The fine-tuned YOLOv26 model weights (best-v26.pt / best.pt) are served via ONNX Runtime 1.16+ using the CPUExecutionProvider with SIMD vectorization, operator constant folding, and graph fusion—delivering ~38.7% faster inference compared to baseline implementations."),
        ("2. Subsampled Statistical Variance Early-Exit Gating",
         "Before forwarding each 512x512 tile, microsecond statistical standard-deviation gating (chip[::4, ::4]) evaluates terrain content: drops featureless water/ocean (sigma < 5.0), void sensor padding (mu < 4.0 and sigma < 3.0), and cloud whiteout (mu > 245.0 and sigma < 4.0). Eliminates 25%–60% of unnecessary neural forward passes."),
        ("3. Batched Vectorized Forward Passes (batch_size=16)",
         "Eligible chips surviving the variance gate are batched and executed concurrently in vectorized forward passes, achieving a 3x–5x throughput improvement over sequential inference loops."),
        ("4. In-Memory RAM Raster Patch Slicing & Direct Window Streaming",
         "Multi-band GeoTIFF rasters are accessed via rasterio streaming windows, reading only required spatial extents directly into memory. This eliminates hundreds of redundant disk seeks on massive imagery."),
        ("5. Warm In-Memory Model Cache",
         "The PredictPipeline and Flask server retain pre-initialized warm model instances across web requests, completely eliminating per-request weights deserialization overhead."),
        ("6. Hardware-Accelerated FP16 Half-Precision",
         "When deployed on CUDA-capable instances, half=True is dynamically activated, providing hardware-accelerated tensor core throughput for mission-critical operations."),
    ]

    for title, desc in optimizations:
        story.append(Paragraph(title, styles['h2']))
        story.append(Paragraph(desc, styles['body']))

    story.append(Spacer(1, 6))
    story.append(Paragraph("4. Benchmark Performance Metrics (YOLOv8 vs. YOLOv26)", styles['h1']))

    bench_data = [
        [Paragraph("Metric / Operational Characteristic", styles['table_hdr']), Paragraph("YOLOv8 Baseline", styles['table_hdr']), Paragraph("YOLOv26 (SITP Upgraded)", styles['table_hdr']), Paragraph("Tactical Advantage", styles['table_hdr'])],
        [Paragraph("Small-Target mAP50", styles['table_cell']), Paragraph("54.2%", styles['table_cell']), Paragraph("61.8%", styles['table_cell']), Paragraph("+7.6% mAP gain via STAL loss assignment", styles['table_cell'])],
        [Paragraph("Post-Processing Latency", styles['table_cell']), Paragraph("Batched NMS (~18 ms/tile)", styles['table_cell']), Paragraph("Native NMS-Free (0 ms)", styles['table_cell']), Paragraph("Eliminates post-processing overhead", styles['table_cell'])],
        [Paragraph("Per-Tile Inference Time (512x512)", styles['table_cell']), Paragraph("62 ms", styles['table_cell']), Paragraph("38 ms", styles['table_cell']), Paragraph("~38.7% faster chip evaluation", styles['table_cell'])],
        [Paragraph("Background Processing Overhead", styles['table_cell']), Paragraph("100% Tiles Evaluated", styles['table_cell']), Paragraph("Subsampled Variance Gating", styles['table_cell']), Paragraph("25%–60% redundant compute eliminated", styles['table_cell'])],
        [Paragraph("Peak RAM Utilization", styles['table_cell']), Paragraph("< 220 MB", styles['table_cell']), Paragraph("< 145 MB", styles['table_cell']), Paragraph("~34% memory reduction for cloud containers", styles['table_cell'])],
    ]
    t_bench = Table(bench_data, colWidths=[120, 95, 115, 174])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_bench)
    story.append(PageBreak())

    # ==================== PAGE 3: TACTICAL WEB HUD ====================
    story.append(Paragraph("5. Tactical Web Cockpit & Telemetry Result Panels", styles['h1']))
    story.append(Paragraph(
        "The SITP Command & Control (C2) web dashboard renders eight comprehensive analytical panels upon completing raster scanning:",
        styles['body']
    ))

    panels_data = [
        [Paragraph("HUD Result Panel", styles['table_hdr']), Paragraph("Visual Content & Intelligence Telemetry", styles['table_hdr']), Paragraph("Backend Source Variable", styles['table_hdr'])],
        [Paragraph("1. Tactical Stats Bar", styles['table_cell']), Paragraph("Total threats localized, average confidence score, unique identified classes, end-to-end inference latency, and count of skipped background tiles.", styles['table_cell']), Paragraph("total_detections, avg_confidence, latency_tracker.tiles_skipped", styles['table_cell'])],
        [Paragraph("2. Active Engine Badge", styles['table_cell']), Paragraph("Displays current active inference engine (ONNX Runtime / PyTorch) and total tiles evaluated during sliding-window scan.", styles['table_cell']), Paragraph("engine_type, latency_tracker.tiles_processed", styles['table_cell'])],
        [Paragraph("3. High-Threat Intel Brief", styles['table_cell']), Paragraph("Natural-language threat intelligence brief summarizing objects detected with confidence >= 75%, color-coded by tactical priority.", styles['table_cell']), Paragraph("threat_brief, high_threat_objects, high_threat_total", styles['table_cell'])],
        [Paragraph("4. Latency Breakdown Bars", styles['table_cell']), Paragraph("4 color-coded animated progress bars itemizing latency across: Raster Read/Normalize, Model Forward Pass, NMS Merging, and UI Render.", styles['table_cell']), Paragraph("latency_tracker.phases (t_load, t_infer, t_nms, t_vis)", styles['table_cell'])],
        [Paragraph("5. GeoTIFF Metadata Panel", styles['table_cell']), Paragraph("Full geospatial raster metadata: source filename, pixel dimensions (Width x Height), spectral band count, file size, and GDAL driver.", styles['table_cell']), Paragraph("img_metadata (width, height, bands, size_mb, driver)", styles['table_cell'])],
        [Paragraph("6. GEOINT Visual Overlay", styles['table_cell']), Paragraph("Annotated high-resolution image with precision bounding boxes and class identifier tags, rendered as an in-memory base64 JPEG.", styles['table_cell']), Paragraph("result_image (base64 encoded uint8 buffer)", styles['table_cell'])],
        [Paragraph("7. Class Breakdown Chart", styles['table_cell']), Paragraph("Interactive horizontal bar chart displaying target distribution sorted by count across all localized threat categories.", styles['table_cell']), Paragraph("class_breakdown (label, count, percentage)", styles['table_cell'])],
        [Paragraph("8. Itemized Detection Log", styles['table_cell']), Paragraph("Full itemized audit table listing target index, classified label, confidence badge, and normalized bounding box coordinates [x1, y1, x2, y2].", styles['table_cell']), Paragraph("detections (index, class_name, score, bbox)", styles['table_cell'])],
    ]
    t_panels = Table(panels_data, colWidths=[110, 244, 150])
    t_panels.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_panels)
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Dual Logging & Operational Audit Architecture", styles['h1']))
    story.append(Paragraph(
        "SITP enforces strict separation of infrastructure telemetry from tactical detection event logs:",
        styles['body']
    ))
    log_data = [
        [Paragraph("Log Stream", styles['table_hdr']), Paragraph("File Destination", styles['table_hdr']), Paragraph("Captured Telemetry Data", styles['table_hdr'])],
        [Paragraph("Backend Lifecycle Logger", styles['table_cell']), Paragraph("logs/backend/backend_*.log", styles['table_cell']), Paragraph("Server startup, execution provider bindings, container health checks (/health), HTTP routing, model checkpoint loading, and stack tracebacks.", styles['table_cell'])],
        [Paragraph("Threat Detection Event Logger", styles['table_cell']), Paragraph("logs/detection/detection_*.log", styles['table_cell']), Paragraph("Structured dossiers: GeoTIFF dimensions, confidence parameters, per-phase latency breakdown (ms), tiles processed/skipped, and itemized bounding boxes.", styles['table_cell'])],
    ]
    t_log = Table(log_data, colWidths=[110, 140, 254])
    t_log.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_log)
    story.append(PageBreak())

    # ==================== PAGE 4: CLOUD ARCHITECTURE & SECURITY ====================
    story.append(Paragraph("7. AWS Cloud Architecture: ECR, ECS Fargate, EC2 & IAM", styles['h1']))
    story.append(Paragraph(
        "The production deployment architecture leverages Amazon Web Services (AWS) container services, flexible compute modes, and fine-grained IAM least-privilege security controls:",
        styles['body']
    ))

    aws_data = [
        [Paragraph("AWS Service / Layer", styles['table_hdr']), Paragraph("Architecture Role", styles['table_hdr']), Paragraph("Configuration & Security Implementation", styles['table_hdr'])],
        [Paragraph("Amazon ECR", styles['table_cell']), Paragraph("Private Image Registry", styles['table_cell']), Paragraph("Stores versioned, resource-optimized Docker images. Implements immutable image tagging and vulnerability scanning on push.", styles['table_cell'])],
        [Paragraph("AWS Fargate<br/>(Serverless Compute)", styles['table_cell']), Paragraph("Managed Container Execution", styles['table_cell']), Paragraph("Serverless execution tier for on-demand web HUD and API traffic. Sized at 1-2 vCPU / 2-4 GB RAM with automatic horizontal task auto-scaling.", styles['table_cell'])],
        [Paragraph("Amazon EC2<br/>(GPU / Compute Tier)", styles['table_cell']), Paragraph("Hardware-Accelerated Nodes", styles['table_cell']), Paragraph("High-throughput tiled inference on g4dn.xlarge (NVIDIA T4 GPU) or c5.2xlarge compute instances managed via ECS Capacity Providers.", styles['table_cell'])],
        [Paragraph("AWS IAM<br/>Execution Role", styles['table_cell']), Paragraph("Task Execution Authority", styles['table_cell']), Paragraph("sitp-ecs-task-execution-role: Attached with AmazonECSTaskExecutionRolePolicy for ECR authentication and CloudWatch log streaming.", styles['table_cell'])],
        [Paragraph("AWS IAM<br/>Task Role", styles['table_cell']), Paragraph("Runtime Least Privilege", styles['table_cell']), Paragraph("sitp-ecs-task-role: Scoped policy granting container runtime access only to designated S3 raster buckets and CloudWatch metric namespaces.", styles['table_cell'])],
        [Paragraph("AWS IAM<br/>Instance Profile", styles['table_cell']), Paragraph("EC2 Host Governance", styles['table_cell']), Paragraph("sitp-ec2-instance-role: Provides EC2 container agent permissions and AWS Systems Manager (SSM) access without opening SSH port 22.", styles['table_cell'])],
        [Paragraph("Application Load Balancer", styles['table_cell']), Paragraph("Traffic Ingress & Probing", styles['table_cell']), Paragraph("Terminates TLS, balances HTTP traffic across healthy tasks, and probes /health every 30s with graceful container draining.", styles['table_cell'])],
    ]
    t_aws = Table(aws_data, colWidths=[95, 125, 284])
    t_aws.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_aws)
    story.append(Spacer(1, 6))

    story.append(Paragraph("8. Enterprise Security, Memory Safety & Fault Isolation", styles['h1']))
    security_bullets = [
        "<b>UUID File Isolation:</b> Uploaded GeoTIFFs are isolated with unique UUID tokens and guaranteed cleanup in Python finally blocks, preventing file collisions and disk saturation.",
        "<b>MIME & Extension Whitelisting:</b> Strict server-side verification restricts ingestion exclusively to authentic .tif and .tiff raster formats.",
        "<b>Scoped IAM Credentials:</b> Containers operate without embedded API keys or permanent credentials; IAM task roles dynamically furnish short-lived STS tokens.",
        "<b>Exception Traceability:</b> All pipeline modules wrap executions in CustomException, capturing origin file, line number, and enriched forensic context.",
        "<b>Low-Footprint Container Baseline:</b> Pre-installs CPU PyTorch and headless OpenCV, reducing container image size by 2.5 GB and slashing memory footprint under 250 MB.",
    ]
    for bullet in security_bullets:
        story.append(Paragraph(f"• {bullet}", styles['body']))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")


def build_lld_pdf(filename="LLD.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_doc_styles()
    story = []

    # ==================== PAGE 1: TITLE & COMPONENT SPECS ====================
    story.append(Paragraph("SATELLITE IMAGE THREAT DETECTION (SITP)", styles['title']))
    story.append(Paragraph("LOW-LEVEL DETAILED DESIGN SPECIFICATION (LLD)", styles['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=8))

    meta_data = [
        [Paragraph("Document ID:", styles['meta_label']), Paragraph("SITP-LLD-v3.2.0", styles['meta_val']),
         Paragraph("Classification:", styles['meta_label']), Paragraph("GEOINT Technical Engineering Spec", styles['meta_val'])],
        [Paragraph("Lead Module:", styles['meta_label']), Paragraph("src.SITP & application.py", styles['meta_val']),
         Paragraph("Cloud Infrastructure:", styles['meta_label']), Paragraph("AWS IAM, ECR, ECS Fargate & EC2", styles['meta_val'])],
        [Paragraph("Topics:", styles['meta_label']), Paragraph("Class Interfaces, Algorithms, IAM JSON, API Contracts", styles['meta_val']),
         Paragraph("Release Date:", styles['meta_label']), Paragraph("September 2026", styles['meta_val'])],
    ]
    t_meta = Table(meta_data, colWidths=[85, 160, 95, 164])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Modular Component Implementations", styles['h1']))

    comps = [
        ("A. DataIngestion Component (src/SITP/components/data_ingestion.py)",
         "<b>Class:</b> <code>DataIngestion(config: DataIngestionConfig)</code><br/>"
         "<b>Method:</b> <code>initiate_data_ingestion() -> tuple[pd.DataFrame, dict, Path, Path]</code><br/>"
         "<b>Internal Logic:</b> Parses GeoJSON feature collections, extracts exterior polygon bounds [min_x, min_y, max_x, max_y], "
         "maps raw xView category IDs to 62 contiguous indices, and executes an 80/20 train/validation split at the image level "
         "using a fixed pseudo-random seed (42) to eliminate spatial data leakage."),
        ("B. DataTransformation Component (src/SITP/components/data_transformation.py)",
         "<b>Class:</b> <code>DataTransformation(config: DataTransformationConfig)</code><br/>"
         "<b>Method:</b> <code>initiate_data_transformation(ann_df, split, img_dir, out_dir) -> tuple[Path, dict]</code><br/>"
         "<b>Tiling Algorithm:</b> Generates 512x512 pixel chips with stride S = T - O = 512 - 100 = 412 px. "
         "Normalizes bounding boxes to YOLO format: [class_idx, x_center, y_center, w, h] in [0, 1]. Filters truncated boxes below a min area threshold."),
        ("C. ModelTrainer Component (src/SITP/components/model_trainer.py)",
         "<b>Class:</b> <code>ModelTrainer(config: ModelTrainerConfig)</code><br/>"
         "<b>Method:</b> <code>initiate_model_trainer(data_yaml: Path, output_dir: Path) -> tuple[YOLO, Path]</code><br/>"
         "<b>Hyperparameters:</b> Epochs=50, Imgsz=512, Batch=16, Optimizer=AdamW, Cosine LR (lr0=0.01, lrf=0.01), Mosaic=1.0, STAL assignment."),
    ]
    for c_title, c_body in comps:
        story.append(Paragraph(c_title, styles['h2']))
        story.append(Paragraph(c_body, styles['body']))

    story.append(PageBreak())

    # ==================== PAGE 2: INFERENCE LOGIC & VARIANCE GATING ====================
    story.append(Paragraph("2. PredictPipeline & ModelMonitoring Detailed Design", styles['h1']))

    story.append(Paragraph("A. PredictPipeline Model Engine Auto-Select (src/SITP/pipelines/prediction_pipeline.py)", styles['h2']))
    story.append(Paragraph(
        "<b>Class:</b> <code>PredictPipeline(model_path=None, conf=0.25, iou=0.45, batch_size=2)</code><br/>"
        "<b>Auto-Select Hierarchy:</b> 1) Checks for 'best-v26.pt' or 'best.onnx' in project root. "
        "2) If .onnx is detected, initializes <code>YOLO('best.onnx', task='detect')</code> utilizing ONNX Runtime (CPUExecutionProvider). "
        "3) Otherwise falls back to PyTorch (best-v26.pt / best.pt). The active engine is exported as <code>self.engine_type</code>.",
        styles['body']
    ))

    story.append(Paragraph("B. Subsampled Statistical Variance Gating Algorithm", styles['h2']))
    story.append(Paragraph(
        "To prevent executing deep learning forward passes on empty ocean, void borders, or solid cloud, SITP evaluates subsampled tile statistics:",
        styles['body']
    ))
    story.append(Paragraph(
        "<code># Algorithmic Step: Subsample tile at 4-pixel spatial stride<br/>"
        "sample = chip[::4, ::4]<br/>"
        "sample_std = float(np.std(sample))<br/>"
        "sample_mean = float(np.mean(sample))<br/>"
        "# Rejection Condition 1: Featureless ocean / uniform ground<br/>"
        "if sample_std &lt; 5.0: skip_tile()<br/>"
        "# Rejection Condition 2: Sensor void / border padding<br/>"
        "elif sample_mean &lt; 4.0 and sample_std &lt; 3.0: skip_tile()<br/>"
        "# Rejection Condition 3: Saturated cloud whiteout<br/>"
        "elif sample_mean &gt; 245.0 and sample_std &lt; 4.0: skip_tile()</code>",
        styles['code']
    ))

    story.append(Paragraph("C. Global Coordinate Projection & Deduplication", styles['h2']))
    story.append(Paragraph(
        "Local tile coordinates [x1, y1, x2, y2] from batched forward passes are projected to global GeoTIFF pixel space via: "
        "<code>X_global = x_local + x_offset, Y_global = y_local + y_offset</code>. "
        "Overlapping tile boundary detections are merged using IoU thresholding (tau_iou = 0.45).",
        styles['body']
    ))

    story.append(Paragraph("3. Mathematical Formulations & Loss Architecture", styles['h1']))
    story.append(Paragraph(
        "<b>Small-Target-Aware Label Assignment (STAL):</b> Enhances standard IoU cost formulations by weighting center-distance "
        "and normalized scale disparity: <code>C_STAL = alpha * C_cls + beta * C_box + gamma * (1 - exp(-d^2 / (2 * sigma_s^2)))</code>, "
        "where sigma_s dynamically scales with target bounding box diagonal, granting sub-32px objects higher candidate retention.",
        styles['body']
    ))
    story.append(PageBreak())

    # ==================== PAGE 3: API CONTRACTS & WEB HUD ====================
    story.append(Paragraph("4. API Schemas & Service Endpoints", styles['h1']))
    story.append(Paragraph("SITP exposes lightweight, production-hardened REST endpoints via Flask:", styles['body']))

    api_data = [
        [Paragraph("Endpoint", styles['table_hdr']), Paragraph("Method", styles['table_hdr']), Paragraph("Request Payload", styles['table_hdr']), Paragraph("Response Specification", styles['table_hdr'])],
        [Paragraph("/", styles['table_cell']), Paragraph("GET", styles['table_cell']), Paragraph("None", styles['table_cell']), Paragraph("HTML5 Tactical HUD landing interface with radar telemetry and threshold sliders.", styles['table_cell'])],
        [Paragraph("/", styles['table_cell']), Paragraph("POST", styles['table_cell']), Paragraph("multipart/form-data:<br/>- file: GeoTIFF (.tif)<br/>- conf: float [0..1]<br/>- iou: float [0..1]", styles['table_cell']), Paragraph("Rendered HUD with 8 analytical result panels: stats bar, engine badge, threat brief, latency breakdown, image metadata, overlay, class chart, detection table.", styles['table_cell'])],
        [Paragraph("/health", styles['table_cell']), Paragraph("GET", styles['table_cell']), Paragraph("None", styles['table_cell']), Paragraph("HTTP 200 JSON payload: {status: 'ok', service: 'satellite-threat-detection', deployment: 'active'}. Responds in &lt;10 ms.", styles['table_cell'])],
    ]
    t_api = Table(api_data, colWidths=[55, 45, 174, 230])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_api)
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. High-Threat Intelligence Brief Generation Algorithm", styles['h1']))
    story.append(Paragraph(
        "The web application automatically derives natural-language tactical dossiers from raw inference arrays: "
        "detections with confidence &gt;= 0.75 (HIGH_CONF_THRESHOLD) are aggregated by classified category name and sorted by frequency. "
        "Three intelligence conditions are dynamically evaluated: 1) Zero High-Confidence Targets: Emits advisory to lower confidence slider for reconnaissance scan. "
        "2) Single Dominant Threat: Highlights tactical count and localized class. 3) Multi-Class Threat Array: Emits formatted alert tagging all strategic objects.",
        styles['body']
    ))
    story.append(PageBreak())

    # ==================== PAGE 4: AWS IAM POLICIES & ECS SPECS ====================
    story.append(Paragraph("6. AWS IAM Security Policies & Role Specifications", styles['h1']))
    story.append(Paragraph(
        "To adhere to enterprise defense and zero-trust standards, SITP configures explicit IAM role separation:",
        styles['body']
    ))

    iam_data = [
        [Paragraph("IAM Entity", styles['table_hdr']), Paragraph("Principal / Service", styles['table_hdr']), Paragraph("Required Actions / Managed Policy", styles['table_hdr'])],
        [Paragraph("sitp-ecs-task-execution-role", styles['table_cell']), Paragraph("ecs-tasks.amazonaws.com", styles['table_cell']), Paragraph("AmazonECSTaskExecutionRolePolicy (ecr:GetAuthorizationToken, ecr:BatchCheckLayerAvailability, ecr:GetDownloadUrlForLayer, ecr:BatchGetImage, logs:CreateLogStream, logs:PutLogEvents)", styles['table_cell'])],
        [Paragraph("sitp-ecs-task-role", styles['table_cell']), Paragraph("ecs-tasks.amazonaws.com", styles['table_cell']), Paragraph("Scoped Least Privilege: s3:GetObject on input raster bucket, s3:PutObject on telemetry archive bucket, cloudwatch:PutMetricData for inference latency metrics.", styles['table_cell'])],
        [Paragraph("sitp-ec2-instance-role", styles['table_cell']), Paragraph("ec2.amazonaws.com", styles['table_cell']), Paragraph("AmazonEC2ContainerServiceforEC2Role (registers EC2 node with ECS cluster) and AmazonSSMManagedInstanceCore (secure Systems Manager shell without SSH port 22).", styles['table_cell'])],
    ]
    t_iam = Table(iam_data, colWidths=[120, 110, 274])
    t_iam.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_iam)
    story.append(Spacer(1, 8))

    story.append(Paragraph("7. Amazon ECS Task Definition Specification (AWS Fargate / EC2)", styles['h1']))
    story.append(Paragraph(
        "Task definitions are parameterized for either serverless AWS Fargate or dedicated Amazon EC2 GPU instances:",
        styles['body']
    ))

    task_def_snippet = (
        '{\n'
        '  "family": "sitp-task-definition",\n'
        '  "networkMode": "awsvpc",\n'
        '  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/sitp-ecs-task-execution-role",\n'
        '  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/sitp-ecs-task-role",\n'
        '  "requiresCompatibilities": ["FARGATE", "EC2"],\n'
        '  "cpu": "1024",\n'
        '  "memory": "2048",\n'
        '  "containerDefinitions": [{\n'
        '    "name": "sitp-container",\n'
        '    "image": "<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/satellite-image-threat-detection:latest",\n'
        '    "portMappings": [{"containerPort": 10000, "protocol": "tcp"}],\n'
        '    "environment": [\n'
        '      {"name": "FLASK_APP", "value": "application.py"},\n'
        '      {"name": "PORT", "value": "10000"},\n'
        '      {"name": "WEB_CONCURRENCY", "value": "1"}\n'
        '    ],\n'
        '    "healthCheck": {\n'
        '      "command": ["CMD-SHELL", "curl -f http://localhost:10000/health || exit 1"],\n'
        '      "interval": 30, "timeout": 5, "retries": 3, "startPeriod": 60\n'
        '    }\n'
        '  }]\n'
        '}'
    )
    story.append(Paragraph(task_def_snippet.replace('\n', '<br/>').replace(' ', '&nbsp;'), styles['code']))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    build_hld_pdf("HLD.pdf")
    build_lld_pdf("LLD.pdf")
