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
        self.result: Optional[ConversionResult] = None
        self.error: Optional[str] = None
        self.params: dict = {}


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
            for path in [job.input_path, job.output_ply_path, job.depth_preview_path]:
                if path and path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass

    # Clean orphaned files in temp dir
    active_paths = set()
    for j in jobs.values():
        if j.status in ("queued", "processing"):
            for p in [j.input_path, j.output_ply_path, j.depth_preview_path]:
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
            )

            job.result = result
            job.status = "complete"
            job.last_updated = time.time()

            # Delete input file immediately after processing
            if job.input_path and job.input_path.exists():
                job.input_path.unlink()

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
