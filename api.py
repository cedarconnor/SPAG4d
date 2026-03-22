# api.py
"""
FastAPI backend for SPAG-4D web UI.
"""

import asyncio
import time
import uuid
from pathlib import Path
import shutil
import subprocess
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from spag4d import SPAG4D, ConversionResult


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
import tempfile
TEMP_DIR = Path(tempfile.gettempdir()) / "spag4d"
JOB_TTL_SECONDS = 30 * 60  # 30 minutes
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
GPU_SEMAPHORE_LIMIT = 1


# ─────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────
processor: Optional[SPAG4D] = None
gpu_semaphore: Optional[asyncio.Semaphore] = None
jobs: dict = {}  # job_id -> JobInfo


class JobInfo:
    """Tracks a conversion job."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "queued"
        self.created_at = time.time()
        self.last_updated = time.time()
        self.input_path: Optional[Path] = None
        self.output_ply_path: Optional[Path] = None
        self.depth_preview_path: Optional[Path] = None
        self.depth_npy_path: Optional[Path] = None
        self.result: Optional[ConversionResult] = None
        self.error: Optional[str] = None
        self.params: dict = {}


class RefineJobInfo:
    """Tracks a refinement job."""

    def __init__(self, refine_id: str, source_job_id: str):
        self.refine_id = refine_id
        self.source_job_id = source_job_id
        self.status = "queued"  # queued, processing, complete, error
        self.created_at = time.time()
        self.last_updated = time.time()
        self.round_number = 0
        self.stage = ""
        self.progress_pct = 0
        self.output_ply_path: Optional[Path] = None
        self.heatmap_ply_path: Optional[Path] = None
        self.diagnostics_dir: Optional[Path] = None
        self.metrics: dict = {}
        self.error: Optional[str] = None
        self.params: dict = {}


refine_jobs: dict = {}  # refine_id -> RefineJobInfo


# ─────────────────────────────────────────────────────────────────
# Lifecycle Management
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global processor, gpu_semaphore

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize with DA360 (default), fall back to DAP, then mock
    try:
        processor = SPAG4D(device="cuda", depth_model="da360")
        print("Loaded DA360 depth model")
    except Exception as e:
        print(f"DA360 not available ({e}), trying DAP...")
        try:
            processor = SPAG4D(device="cuda", depth_model="dap")
            print("Loaded DAP depth model")
        except Exception as e2:
            print(f"DAP not available ({e2}), using mock depth")
            processor = SPAG4D(device="cuda", use_mock_dap=True)

    gpu_semaphore = asyncio.Semaphore(GPU_SEMAPHORE_LIMIT)
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    await run_cleanup()


async def cleanup_loop():
    """Periodic cleanup of expired jobs and temp files."""
    while True:
        try:
            await asyncio.sleep(60)
            await run_cleanup()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Cleanup error: {e}")
            await asyncio.sleep(60)


async def run_cleanup():
    """Remove expired jobs and their files."""
    now = time.time()

    expired_jobs = [
        job_id for job_id, job in jobs.items()
        if job.status in ("complete", "error")
        and now - job.last_updated > JOB_TTL_SECONDS
    ]

    for job_id in expired_jobs:
        job = jobs.pop(job_id, None)
        if job:
            for path in [job.input_path, job.output_ply_path, job.depth_preview_path, job.depth_npy_path]:
                if path and path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass

    # Clean orphaned files in temp dir
    active_paths = set()
    for j in jobs.values():
        if j.status in ("queued", "processing"):
            for p in [j.input_path, j.output_ply_path, j.depth_preview_path, j.depth_npy_path]:
                if p:
                    active_paths.add(str(p))

    try:
        for f in TEMP_DIR.iterdir():
            if str(f) in active_paths:
                continue
            if now - f.stat().st_mtime > JOB_TTL_SECONDS:
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink()
    except Exception:
        pass


app = FastAPI(title="SPAG-4D", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────
# COOP/COEP Middleware (required for SharedArrayBuffer)
# ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_coop_coep(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


# ─────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────
@app.post("/api/convert")
async def convert_panorama(
    file: UploadFile = File(...),
    depth_model: str = Query("da360", pattern="^(dap|da360)$"),
    sharp_refine: bool = Query(False),
    stride: int = Query(2, ge=1, le=8),
    depth_min: float = Query(0.1, ge=0.01),
    depth_max: float = Query(100.0, le=1000.0),
    sky_threshold: float = Query(80.0),
    outlier_pruning: float = Query(0.0, ge=0.0, le=1.0),
    global_scale: float = Query(1.0, ge=0.1, le=10.0),
    sharp_projection: str = Query("icosahedral"),
    sharp_cubemap_size: int = Query(1536, ge=256, le=4096),
):
    """Convert uploaded panorama to Gaussian splat PLY."""
    if sharp_projection not in ("cubemap", "icosahedral"):
        raise HTTPException(400, f"Invalid sharp_projection: {sharp_projection!r}. Must be 'cubemap' or 'icosahedral'.")
    if depth_min >= depth_max:
        raise HTTPException(400, f"depth_min ({depth_min}) must be less than depth_max ({depth_max}).")

    # Stream upload with incremental size check
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"File too large. Max: {MAX_UPLOAD_SIZE // 1024 // 1024}MB")
        chunks.append(chunk)
    content = b"".join(chunks)

    job_id = str(uuid.uuid4())
    job = JobInfo(job_id)
    jobs[job_id] = job

    job.params = {
        "depth_model": depth_model,
        "sharp_refine": sharp_refine,
        "stride": stride,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "sky_threshold": sky_threshold,
        "outlier_pruning": outlier_pruning,
        "global_scale": global_scale,
        "sharp_projection": sharp_projection,
        "sharp_cubemap_size": sharp_cubemap_size,
    }

    suffix = Path(file.filename).suffix if file.filename else '.jpg'
    job.input_path = TEMP_DIR / f"{job_id}_input{suffix}"
    job.output_ply_path = TEMP_DIR / f"{job_id}_output.ply"
    job.depth_preview_path = TEMP_DIR / f"{job_id}_depth.jpg"
    job.depth_npy_path = TEMP_DIR / f"{job_id}_depth.npy"

    with open(job.input_path, "wb") as f:
        f.write(content)

    asyncio.create_task(process_job(
        job,
        depth_model=depth_model,
        sharp_refine=sharp_refine,
        stride=stride,
        depth_min=depth_min,
        depth_max=depth_max,
        sky_threshold=sky_threshold,
        outlier_pruning=outlier_pruning,
        global_scale=global_scale,
        sharp_projection=sharp_projection,
        sharp_cubemap_size=sharp_cubemap_size,
    ))

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "queue_position": sum(1 for j in jobs.values() if j.status == "queued"),
    })


async def process_job(
    job: JobInfo,
    depth_model: str = "dap",
    sharp_refine: bool = False,
    stride: int = 2,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    sky_threshold: float = 80.0,
    outlier_pruning: float = 0.0,
    global_scale: float = 1.0,
    sharp_projection: str = "icosahedral",
    sharp_cubemap_size: int = 1536,
):
    """Process conversion job with GPU semaphore."""
    try:
        job.status = "queued"
        async with gpu_semaphore:
            job.status = "processing"
            job.last_updated = time.time()

            result = await run_in_threadpool(
                processor.convert,
                input_path=str(job.input_path),
                output_path=str(job.output_ply_path),
                depth_min=depth_min,
                depth_max=depth_max,
                sky_threshold=sky_threshold,
                stride=stride,
                outlier_pruning=outlier_pruning,
                global_scale=global_scale,
                depth_model=depth_model,
                sharp_refine=sharp_refine,
                sharp_projection=sharp_projection,
                sharp_cubemap_size=sharp_cubemap_size,
                depth_preview_path=str(job.depth_preview_path),
                depth_npy_path=str(job.depth_npy_path),
            )

            job.result = result
            job.status = "complete"
            job.last_updated = time.time()

            # Keep input panorama for potential refinement (cleaned up with job expiry)

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "error"
        job.error = str(e)
        print(f"Job failed with error: {e}")
        job.last_updated = time.time()


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job.status,
    }

    if job.status == "queued":
        response["queue_position"] = sum(
            1 for j in jobs.values()
            if j.status == "queued" and j.created_at < job.created_at
        ) + 1

    if job.status == "complete" and job.result:
        response["splat_count"] = job.result.splat_count
        response["file_size_mb"] = round(job.result.file_size / 1024 / 1024, 2)
        response["processing_time"] = round(job.result.processing_time, 2)
        response["ply_url"] = f"/api/download/{job_id}"
        if job.depth_preview_path and job.depth_preview_path.exists():
            response["depth_preview_url"] = f"/api/depth_preview/{job_id}"
        # Refinement readiness: need PLY, panorama, and depth
        has_ply = job.output_ply_path and job.output_ply_path.exists()
        has_pano = job.input_path and job.input_path.exists()
        has_depth = job.depth_npy_path and job.depth_npy_path.exists()
        response["refineable"] = bool(has_ply and has_pano and has_depth)

    if job.status == "error":
        response["error"] = job.error

    if job.params:
        response["params"] = job.params

    return JSONResponse(response)


@app.get("/api/depth_preview/{job_id}")
async def get_depth_preview(job_id: str):
    """Get depth map preview image."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Job not complete")

    if not job.depth_preview_path or not job.depth_preview_path.exists():
        raise HTTPException(404, "Depth preview not available")

    return FileResponse(
        job.depth_preview_path,
        media_type="image/jpeg",
        filename=f"depth_{job_id[:8]}.jpg"
    )


@app.get("/api/download/{job_id}")
async def download_file(job_id: str):
    """Download the generated PLY file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Job not complete")

    if not job.output_ply_path or not job.output_ply_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(
        job.output_ply_path,
        media_type="application/octet-stream",
        filename=f"spag4d_{job_id[:8]}.ply"
    )


# ─────────────────────────────────────────────────────────────────
# Refinement Endpoints
# ─────────────────────────────────────────────────────────────────
@app.post("/api/refine")
async def start_refinement(
    job_id: str = Query(..., description="Source conversion job ID"),
    orbit_radius: float = Query(0.5, ge=0.05, le=5.0),
    n_cameras: int = Query(8, ge=2, le=32),
    max_rounds: int = Query(2, ge=1, le=5),
    synthesis_backend: str = Query("klein-sharp"),
    camera_preset: str = Query("orbit"),
    custom_cameras: Optional[str] = Query(None, description="JSON array of custom camera poses"),
):
    """Start refinement on an existing conversion job."""
    if job_id not in jobs:
        raise HTTPException(404, "Source job not found")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Source job not complete")

    # Verify all artifacts exist
    if not (job.output_ply_path and job.output_ply_path.exists()):
        raise HTTPException(400, "PLY file not found")
    if not (job.input_path and job.input_path.exists()):
        raise HTTPException(400, "Input panorama not found (may have been cleaned up)")
    if not (job.depth_npy_path and job.depth_npy_path.exists()):
        raise HTTPException(400, "Depth map not found")

    refine_id = str(uuid.uuid4())
    refine_job = RefineJobInfo(refine_id, job_id)
    refine_job.params = {
        "orbit_radius": orbit_radius,
        "n_cameras": n_cameras,
        "max_rounds": max_rounds,
        "synthesis_backend": synthesis_backend,
        "camera_preset": camera_preset,
        "custom_cameras": custom_cameras,
    }
    refine_job.output_ply_path = TEMP_DIR / f"{refine_id}_refined.ply"
    refine_job.heatmap_ply_path = TEMP_DIR / f"{refine_id}_heatmap.ply"
    refine_job.diagnostics_dir = TEMP_DIR / f"{refine_id}_diagnostics"
    refine_jobs[refine_id] = refine_job

    asyncio.create_task(process_refinement(refine_job, job))

    return JSONResponse({
        "refine_job_id": refine_id,
        "status": "queued",
    })


async def process_refinement(refine_job: RefineJobInfo, source_job: JobInfo):
    """Run refinement pipeline in background."""
    try:
        async with gpu_semaphore:
            refine_job.status = "processing"
            refine_job.last_updated = time.time()

            result = await run_in_threadpool(
                _run_refinement,
                source_job=source_job,
                refine_job=refine_job,
            )

            refine_job.metrics = result
            refine_job.status = "complete"
            refine_job.last_updated = time.time()

    except Exception as e:
        import traceback
        traceback.print_exc()
        refine_job.status = "error"
        refine_job.error = str(e)
        refine_job.last_updated = time.time()


def _run_refinement(source_job: JobInfo, refine_job: RefineJobInfo) -> dict:
    """Execute the refinement pipeline (blocking, runs in thread)."""
    import sys
    _refine_path = str(Path(__file__).resolve().parent / "spag4d-refine")
    if _refine_path not in sys.path:
        sys.path.insert(0, _refine_path)

    from spag4d_refine.config import RefineConfig
    from spag4d_refine.pipeline import RefinePipeline

    params = refine_job.params
    output_dir = refine_job.diagnostics_dir or TEMP_DIR / f"{refine_job.refine_id}_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RefineConfig(
        splat_path=source_job.output_ply_path,
        panorama_rgb_path=source_job.input_path,
        panorama_depth_path=source_job.depth_npy_path,
        output_dir=output_dir,
        max_refinement_rounds=params.get("max_rounds", 2),
        synthesis_backend=params.get("synthesis_backend", "klein-sharp"),
        orbit_radius=params.get("orbit_radius", 0.5),
        n_cameras=params.get("n_cameras", 8),
        camera_preset=params.get("camera_preset", "orbit"),
    )

    # Build custom cameras if provided
    custom_camera_set = None
    custom_cameras_json = params.get("custom_cameras")
    if params.get("camera_preset") == "custom" and custom_cameras_json:
        import json as json_mod
        from spag4d_refine.camera.pinhole import PinholeCamera, CameraSet
        import numpy as np
        poses = json_mod.loads(custom_cameras_json)
        cams = []
        for p in poses:
            cam = PinholeCamera.look_at(
                eye=np.array(p["position"]),
                target=np.array(p["target"]),
                up=np.array(p.get("up", [0, 1, 0])),
                vfov_deg=config.camera_vfov_deg,
                width=config.render_resolution[0],
                height=config.render_resolution[1],
            )
            cams.append(cam)
        custom_camera_set = CameraSet(cams)

    def update_progress(round_num, stage, pct):
        refine_job.round_number = round_num
        refine_job.stage = stage
        refine_job.progress_pct = pct
        refine_job.last_updated = time.time()

    pipeline = RefinePipeline(config)
    result = pipeline.run(
        progress_callback=update_progress,
        cameras=custom_camera_set,
    )

    # Copy final PLY and heatmap to expected locations
    import shutil
    if result.output_path.exists():
        shutil.copy2(result.output_path, refine_job.output_ply_path)
    if result.heatmap_path and result.heatmap_path.exists():
        shutil.copy2(result.heatmap_path, refine_job.heatmap_ply_path)

    return {
        "original_count": result.original_gaussian_count,
        "final_count": result.final_gaussian_count,
        "gaussians_added": result.gaussians_added,
        "gaussians_pruned": result.gaussians_pruned,
        "rounds_completed": result.rounds_completed,
        "total_time": round(result.total_elapsed_seconds, 1),
        "synthesis_backend": result.synthesis_backend_used,
        "round_stats": [
            {
                "round": s.round_number,
                "seeded": s.gaussians_seeded,
                "promoted": s.gaussians_promoted,
                "pruned": s.gaussians_pruned,
                "psnr": round(s.original_psnr, 2),
                "time": round(s.elapsed_seconds, 1),
            }
            for s in result.round_stats
        ],
    }


@app.get("/api/refine/status/{refine_id}")
async def get_refine_status(refine_id: str):
    """Get refinement job status."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")

    rj = refine_jobs[refine_id]
    response = {
        "refine_job_id": refine_id,
        "status": rj.status,
        "round": rj.round_number,
        "stage": rj.stage,
        "progress_pct": rj.progress_pct,
    }

    # Diagnostics available during processing and after completion
    if rj.diagnostics_dir and rj.diagnostics_dir.exists():
        response["diagnostics_url"] = f"/api/refine/diagnostics/{refine_id}"

    if rj.status == "complete":
        response["metrics"] = rj.metrics
        if rj.output_ply_path and rj.output_ply_path.exists():
            response["ply_url"] = f"/api/refine/download/{refine_id}"
        if rj.heatmap_ply_path and rj.heatmap_ply_path.exists():
            response["heatmap_url"] = f"/api/refine/heatmap/{refine_id}"

    if rj.status == "error":
        response["error"] = rj.error

    return JSONResponse(response)


@app.get("/api/refine/download/{refine_id}")
async def download_refined_ply(refine_id: str):
    """Download refined PLY file."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")

    rj = refine_jobs[refine_id]
    if rj.status != "complete":
        raise HTTPException(400, "Refinement not complete")

    if not rj.output_ply_path or not rj.output_ply_path.exists():
        raise HTTPException(404, "Refined PLY not found")

    return FileResponse(
        rj.output_ply_path,
        media_type="application/octet-stream",
        filename=f"refined_{refine_id[:8]}.ply",
    )


@app.get("/api/refine/heatmap/{refine_id}")
async def download_heatmap_ply(refine_id: str):
    """Download provenance heatmap PLY file."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")
    rj = refine_jobs[refine_id]
    if not rj.heatmap_ply_path or not rj.heatmap_ply_path.exists():
        raise HTTPException(404, "Heatmap PLY not found")
    return FileResponse(
        rj.heatmap_ply_path,
        media_type="application/octet-stream",
        filename=f"heatmap_{refine_id[:8]}.ply",
    )


@app.get("/api/refine/diagnostics/{refine_id}")
async def get_refine_diagnostics(refine_id: str):
    """List diagnostic images for a refinement job."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")

    rj = refine_jobs[refine_id]
    # Diagnostics are saved in output_dir/diagnostics/ subdirectory
    diag_dir = rj.diagnostics_dir / "diagnostics" if rj.diagnostics_dir else None
    if not diag_dir or not diag_dir.exists():
        # Fallback to root diagnostics dir
        diag_dir = rj.diagnostics_dir
    if not diag_dir or not diag_dir.exists():
        raise HTTPException(404, "No diagnostics available")

    images = sorted(
        f.name for f in diag_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    # Group by camera for the preview gallery
    cameras = {}
    for name in images:
        # Parse r{round}_cam{idx}_{type}.png
        parts = name.replace(".png", "").replace(".jpg", "").split("_")
        cam_key = "_".join(parts[:2]) if len(parts) >= 3 else name
        img_type = parts[2] if len(parts) >= 3 else "combined"
        if cam_key not in cameras:
            cameras[cam_key] = {}
        cameras[cam_key][img_type] = f"/api/refine/diagnostics/{refine_id}/{name}"

    return JSONResponse({
        "refine_job_id": refine_id,
        "cameras": cameras,
        "images": [
            {"name": name, "url": f"/api/refine/diagnostics/{refine_id}/{name}"}
            for name in images
        ],
    })


@app.get("/api/refine/diagnostics/{refine_id}/{filename}")
async def get_refine_diagnostic_image(refine_id: str, filename: str):
    """Serve a single diagnostic image."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")

    rj = refine_jobs[refine_id]
    if not rj.diagnostics_dir:
        raise HTTPException(404, "No diagnostics available")

    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    # Check in diagnostics/ subdirectory first, then root
    image_path = rj.diagnostics_dir / "diagnostics" / safe_name
    if not image_path.exists():
        image_path = rj.diagnostics_dir / safe_name
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(404, "Diagnostic image not found")

    media_type = "image/png" if safe_name.endswith(".png") else "image/jpeg"
    return FileResponse(image_path, media_type=media_type)


@app.get("/api/refine/metrics/{refine_id}")
async def get_refine_metrics(refine_id: str):
    """Get refinement metrics."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")

    rj = refine_jobs[refine_id]
    if rj.status != "complete":
        raise HTTPException(400, "Refinement not complete")

    return JSONResponse(rj.metrics)


@app.post("/api/shutdown")
async def shutdown_server(request: Request):
    """Shut down the SPAG-4D server (localhost only)."""
    import os

    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "Shutdown is only allowed from localhost")

    async def _exit():
        await asyncio.sleep(0.5)
        os._exit(0)

    asyncio.create_task(_exit())
    return JSONResponse({"status": "shutting_down"})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_available": gpu_semaphore._value > 0 if gpu_semaphore else False,
        "active_jobs": sum(1 for j in jobs.values() if j.status == "processing"),
        "queued_jobs": sum(1 for j in jobs.values() if j.status == "queued"),
    }


# Serve test images
TEST_IMAGE_DIR = Path("./TestImage")
if TEST_IMAGE_DIR.exists():
    app.mount("/TestImage", StaticFiles(directory="TestImage"), name="test-images")

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ─────────────────────────────────────────────────────────────────
# Startup: kill any existing server on the target port
# ─────────────────────────────────────────────────────────────────
def kill_existing_server(port: int):
    """Kill any existing SPAG-4D server on *port* before we bind."""
    import urllib.request

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shutdown", method="POST"
        )
        urllib.request.urlopen(req, timeout=2)
        time.sleep(1)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if "LISTENING" in line and len(parts) >= 2:
                local_addr = parts[1]
                if local_addr.endswith(f":{port}"):
                    pid = parts[-1]
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True,
                    )
    except Exception:
        pass


DEFAULT_PORT = 7860


if __name__ == "__main__":
    import argparse, uvicorn

    parser = argparse.ArgumentParser(description="SPAG-4D server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    kill_existing_server(args.port)
    uvicorn.run("api:app", host=args.host, port=args.port)
