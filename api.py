# api.py
"""
FastAPI backend for SPAG-4D web UI.
"""

import asyncio
import time
import uuid
from pathlib import Path
import shutil
import zipfile
import subprocess
import cv2
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from spag4d import SPAG4D, ConversionResult


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
import tempfile
# Use system temp dir to avoid permission issues/clutter
TEMP_DIR = Path(tempfile.gettempdir()) / "spag4d"
JOB_TTL_SECONDS = 30 * 60  # 30 minutes
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
GPU_SEMAPHORE_LIMIT = 1  # Only 1 concurrent GPU job


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
        self.last_updated = time.time()  # Track last activity
        self.input_path: Optional[Path] = None
        self.output_ply_path: Optional[Path] = None
        self.output_splat_path: Optional[Path] = None
        self.preview_splat_path: Optional[Path] = None
        self.depth_preview_path: Optional[Path] = None
        self.result: Optional[ConversionResult] = None
        self.error: Optional[str] = None
        self.params: dict = {}  # Store conversion params for UI feedback
        
        # Video specific
        self.is_video = False
        self.total_frames = 0
        self.current_frame = 0
        self.frame_manifest: list = []
        self.frames_dir: Optional[Path] = None
        self.output_zip_path: Optional[Path] = None


# ─────────────────────────────────────────────────────────────────
# Lifecycle Management
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global processor, gpu_semaphore
    
    # Startup
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to initialize with PanDA (preferred), fall back to DAP, then mock
    try:
        processor = SPAG4D(device="cuda", depth_model="panda")
        print("Loaded PanDA depth model")
    except Exception as e:
        print(f"PanDA model not available ({e}), trying DAP...")
        try:
            processor = SPAG4D(device="cuda", depth_model="dap")
            print("Loaded DAP depth model")
        except Exception as e2:
            print(f"DAP model not available ({e2}), using mock depth")
            processor = SPAG4D(device="cuda", depth_model="mock")
    
    gpu_semaphore = asyncio.Semaphore(GPU_SEMAPHORE_LIMIT)
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_loop())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    await run_cleanup()


async def cleanup_loop():
    """Periodic cleanup of expired jobs and temp files."""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            await run_cleanup()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Cleanup error: {e}")
            await asyncio.sleep(60)  # Wait before retry


async def run_cleanup():
    """Remove expired jobs and their files."""
    now = time.time()
    
    # Only cleanup completed or errored jobs that are older than TTL
    # Never cleanup queued or processing jobs based on time
    expired_jobs = [
        job_id for job_id, job in jobs.items()
        if job.status in ("complete", "error") 
        and now - job.last_updated > JOB_TTL_SECONDS
    ]
    
    for job_id in expired_jobs:
        job = jobs.pop(job_id, None)
        if job:
            # Delete associated files
            for path in [job.input_path, job.output_ply_path, 
                        job.output_splat_path, job.preview_splat_path,
                        job.frames_dir, job.output_zip_path]:
                if path and path.exists():
                    try:
                        if path.is_dir():
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            path.unlink()
                    except Exception:
                        pass
    
    # Also clean orphaned files in temp dir
    # Collect paths of active jobs to avoid deleting their files
    active_paths = set()
    for j in jobs.values():
        if j.status in ("queued", "processing"):
            for p in [j.input_path, j.output_ply_path, j.output_splat_path,
                      j.preview_splat_path, j.depth_preview_path,
                      j.frames_dir, j.output_zip_path]:
                if p:
                    active_paths.add(str(p))
    
    try:
        for f in TEMP_DIR.iterdir():
            if str(f) in active_paths:
                continue  # Skip files belonging to active jobs
            if now - f.stat().st_mtime > JOB_TTL_SECONDS:
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink()
    except Exception:
        pass


app = FastAPI(title="SPAG-4D", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────
@app.post("/api/convert")
async def convert_panorama(
    file: UploadFile = File(...),
    stride: int = Query(2, ge=1, le=8),
    scale_factor: float = Query(1.5, ge=0.1, le=5.0),
    thickness: float = Query(0.1, ge=0.01, le=1.0),
    global_scale: float = Query(1.0, ge=0.1, le=10.0),
    depth_min: float = Query(0.1, ge=0.01),

    depth_max: float = Query(100.0, le=1000.0),
    # SHARP params (enabled by default for maximum quality)
    sharp_refine: bool = Query(True),
    sharp_projection: str = Query("cubemap"),  # "cubemap" or "icosahedral"
    scale_blend: float = Query(0.8, ge=0.0, le=1.0),
    color_blend: float = Query(0.5, ge=0.0, le=1.0),
    opacity_blend: float = Query(1.0, ge=0.0, le=1.0),
    sharp_cubemap_size: int = Query(1536),
    sky_threshold: float = Query(80.0),
    sky_dome: bool = Query(True),
    # Depth model selection
    depth_model: str = Query("panda"),
    da3_projection: str = Query("equirectangular"),
    guided_filter: bool = Query(True),
    guided_strength: float = Query(0.25, ge=0.0, le=1.0)
):
    """
    Convert uploaded panorama to Gaussian splat.
    
    Returns job_id immediately. Poll /api/status/{job_id} for progress.
    """
    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max: {MAX_UPLOAD_SIZE // 1024 // 1024}MB")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobInfo(job_id)
    jobs[job_id] = job
    
    # Force strength to 0 if filter is disabled (prevents state leak from reused processor)
    if not guided_filter:
        guided_strength = 0.0
    
    # Store params for feedback
    job.params = {
        "depth_model": depth_model,
        "da3_projection": da3_projection,
        "guided_filter": guided_filter,
        "guided_strength": guided_strength,
        "sharp_refine": sharp_refine,
        "sharp_projection": sharp_projection,
        "scale_blend": scale_blend,
        "color_blend": color_blend,
        "opacity_blend": opacity_blend,
        "sharp_cubemap_size": sharp_cubemap_size,
        "sky_threshold": sky_threshold,
        "sky_dome": sky_dome
    }
    
    # Re-initialize processor if depth model changed
    global processor
    if depth_model != processor.depth_model_name:
        try:
            processor = SPAG4D(
                device="cuda",
                depth_model=depth_model,
                use_guided_filter=guided_filter,
            )
            print(f"Switched to depth model: {depth_model}")
        except Exception as e:
            print(f"Failed to switch depth model to {depth_model}: {e}")
            print(f"Falling back to current model: {processor.depth_model_name}")
            depth_model = processor.depth_model_name

    # Determine file extension
    suffix = Path(file.filename).suffix if file.filename else '.jpg'
    
    # Save input file
    job.input_path = TEMP_DIR / f"{job_id}_input{suffix}"
    job.output_ply_path = TEMP_DIR / f"{job_id}_output.ply"
    job.output_splat_path = TEMP_DIR / f"{job_id}_output.splat"
    job.preview_splat_path = TEMP_DIR / f"{job_id}_preview.splat"
    job.depth_preview_path = TEMP_DIR / f"{job_id}_depth.jpg"
    
    with open(job.input_path, "wb") as f:
        f.write(content)
    
    # Queue processing (non-blocking)
    asyncio.create_task(process_job(
        job, stride, scale_factor, thickness, global_scale, depth_min, depth_max,
        sharp_refine, sharp_projection, scale_blend, opacity_blend,
        sharp_cubemap_size, sky_threshold, color_blend=color_blend,
        sky_dome=sky_dome, da3_projection=da3_projection
    ))
    
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "queue_position": sum(1 for j in jobs.values() if j.status == "queued")
    })


@app.post("/api/convert_video")
async def convert_video(
    file: UploadFile = File(...),
    fps: int = Query(10, ge=1, le=60),
    stride: int = Query(2, ge=1, le=8),
    scale_factor: float = Query(1.5, ge=0.1, le=5.0),
    thickness: float = Query(0.1, ge=0.01, le=1.0),
    global_scale: float = Query(1.0, ge=0.1, le=10.0),
    depth_min: float = Query(0.1, ge=0.01),
    depth_max: float = Query(100.0, le=1000.0),
    start_time: float = Query(0.0, ge=0.0),
    duration: Optional[float] = Query(None, gt=0.0),
    temporal_alpha: float = Query(0.3, ge=0.0, le=1.0),
    stabilize_video: bool = Query(False),
    color_blend: float = Query(0.5, ge=0.0, le=1.0),
    opacity_blend: float = Query(1.0, ge=0.0, le=1.0),
    sharp_cubemap_size: int = Query(1536),
    sky_threshold: float = Query(80.0),
    sky_dome: bool = Query(True),
    depth_model: str = Query("panda"),
    da3_projection: str = Query("equirectangular"),
):
    """Convert uploaded 360 video to sequence of Gaussian splats."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE * 5: # Allow larger upload for video
        raise HTTPException(400, f"File too large.")
    
    job_id = str(uuid.uuid4())
    job = JobInfo(job_id)
    job.is_video = True
    jobs[job_id] = job
    
    # Re-initialize processor if depth model changed
    global processor
    if depth_model != processor.depth_model_name:
        try:
            processor = SPAG4D(
                device="cuda",
                depth_model=depth_model,
                use_guided_filter=True, # Video doesn't have a UI toggle for this, assume true
            )
            print(f"Switched to depth model: {depth_model}")
        except Exception as e:
            print(f"Failed to switch depth model to {depth_model}: {e}")
            print(f"Falling back to current model: {processor.depth_model_name}")
            depth_model = processor.depth_model_name

    # Save input video
    suffix = Path(file.filename).suffix if file.filename else '.mp4'
    job.input_path = TEMP_DIR / f"{job_id}_input{suffix}"
    job.output_zip_path = TEMP_DIR / f"{job_id}_sequence.zip"
    
    with open(job.input_path, "wb") as f:
        f.write(content)
    
    asyncio.create_task(process_video_job(
        job, fps, stride, scale_factor, thickness, global_scale, depth_min, depth_max,
        sky_threshold,
        start_time, duration, temporal_alpha, stabilize_video, scale_blend, opacity_blend,
        sharp_cubemap_size, color_blend, da3_projection=da3_projection
    ))
    
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "queue_position": sum(1 for j in jobs.values() if j.status == "queued")
    })


async def process_job(
    job: JobInfo,
    stride: int,
    scale_factor: float,
    thickness: float,
    global_scale: float,

    depth_min: float,
    depth_max: float,
    sharp_refine: bool = True,
    sharp_projection: str = "cubemap",
    scale_blend: float = 0.8,
    opacity_blend: float = 1.0,
    sharp_cubemap_size: int = 1536,
    sky_threshold: float = 80.0,
    color_blend: float = 0.5,
    sky_dome: bool = True,
    da3_projection: str = "equirectangular",
    guided_strength: float = 0.25
):
    """Process conversion job with GPU semaphore."""
    global processor
    
    try:
        # Wait for GPU access
        job.status = "queued"
        async with gpu_semaphore:
            job.status = "processing"
            
            # Auto-adjust stride for high-resolution images to prevent OOM
            from PIL import Image
            with Image.open(job.input_path) as img:
                width, height = img.size
            
            effective_stride = stride
            # panda/dap process the full ERP in one pass — cap stride at 4 on 8K+ to
            # avoid OOM.  DA3 projects into face tiles so it handles resolution itself.
            if processor.depth_model_name in ("panda", "dap"):
                if width > 6000:
                    effective_stride = max(stride, 4)
                    print(f"⚠️ High-res image ({width}x{height}), using stride={effective_stride}")
                elif width > 4096:
                    effective_stride = max(stride, 2)
                    print(f"⚠️ Large image ({width}x{height}), using stride={effective_stride}")
            
            # Store adjusted stride in params for user feedback
            if effective_stride != stride:
                job.params["adjusted_stride"] = effective_stride
                job.params["original_stride"] = stride
            
            # Run heavy computation in thread pool (doesn't block event loop)
            result = await run_in_threadpool(
                processor.convert,
                input_path=str(job.input_path),
                output_path=str(job.output_ply_path),
                stride=effective_stride,
                scale_factor=scale_factor,
                thickness_ratio=thickness,
                global_scale=global_scale,
                depth_min=depth_min,
                depth_max=depth_max,
                depth_preview_path=str(job.depth_preview_path),
                use_sharp_refinement=sharp_refine,
                scale_blend=scale_blend,
                opacity_blend=opacity_blend,
                color_blend=color_blend,
                sharp_cubemap_size=sharp_cubemap_size,
                sky_threshold=sky_threshold,
                sky_dome=sky_dome,
                da3_projection=da3_projection,
                guided_strength=guided_strength
            )
            
            # Generate web preview (low-res SPLAT)
            await run_in_threadpool(
                processor.convert,
                input_path=str(job.input_path),
                output_path=str(job.preview_splat_path),
                stride=8,  # Always use stride 8 for preview
                scale_factor=scale_factor,
                thickness_ratio=thickness,
                global_scale=global_scale,
                depth_min=depth_min,
                depth_max=depth_max,
                output_format="splat",
                use_sharp_refinement=sharp_refine,
                scale_blend=scale_blend,
                opacity_blend=opacity_blend,
                color_blend=color_blend,
                sharp_cubemap_size=sharp_cubemap_size,
                sky_threshold=sky_threshold,
                da3_projection=da3_projection,
                sky_dome=sky_dome
            )
            
            # Also generate full SPLAT for download
            await run_in_threadpool(
                processor.convert_ply_to_splat,
                str(job.output_ply_path),
                str(job.output_splat_path)
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


async def process_video_job(
    job: JobInfo,
    fps: int,
    stride: int,
    scale_factor: float,
    thickness: float,
    global_scale: float,
    depth_min: float,
    depth_max: float,
    sky_threshold: float = 80.0,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    temporal_alpha: float = 0.7,
    stabilize_video: bool = False,
    scale_blend: float = 0.5,
    opacity_blend: float = 1.0,
    sharp_cubemap_size: int = 1536,
    color_blend: float = 0.5,
    da3_projection: str = "equirectangular",
):
    """Process video conversion with batched inference, temporal smoothing, and stabilization."""
    global processor
    
    import torch
    import numpy as np
    from PIL import Image
    
    BATCH_SIZE = 4  # Process 4 frames at a time
    
    # Initialize visual odometry if stabilization enabled
    vo = None
    if stabilize_video:
        from spag4d.visual_odometry import SphericalVisualOdometry, transform_gaussians
        vo = SphericalVisualOdometry(n_features=1000)
    
    try:
        job.status = "queued"
        async with gpu_semaphore:
            job.status = "processing"
            job.last_updated = time.time()
            
            # 1. Extract frames
            frames_dir = TEMP_DIR / f"{job.job_id}_frames"
            job.frames_dir = frames_dir
            frames_dir.mkdir(exist_ok=True)
            
            # Run ffmpeg to extract frames
            cmd = ['ffmpeg']
            if start_time > 0:
                cmd.extend(['-ss', str(start_time)])
            if duration:
                cmd.extend(['-t', str(duration)])
            cmd.extend([
                '-i', str(job.input_path),
                '-vf', f'fps={fps}',
                '-qscale:v', '2',
                str(frames_dir / 'frame_%05d.jpg')
            ])
            
            await run_in_threadpool(subprocess.run, cmd, check=True, capture_output=True)
            
            frames = sorted(list(frames_dir.glob('frame_*.jpg')))
            job.total_frames = len(frames)
            
            if job.total_frames == 0:
                raise Exception("No frames extracted from video")
            
            output_dir = TEMP_DIR / f"{job.job_id}_output"
            output_dir.mkdir(exist_ok=True)
            
            # ─────────────────────────────────────────────────────────────
            # PASS 1: Batched Depth Inference
            # ─────────────────────────────────────────────────────────────
            depths = []
            masks = []
            
            for batch_start in range(0, len(frames), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(frames))
                batch_frames = frames[batch_start:batch_end]
                
                # Load batch of images
                batch_tensors = []
                for frame_path in batch_frames:
                    img = Image.open(frame_path).convert('RGB')
                    img_np = np.array(img)
                    img_tensor = torch.from_numpy(img_np).to(processor.device)
                    batch_tensors.append(img_tensor)
                
                # Stack into batch [B, H, W, 3]
                batch = torch.stack(batch_tensors, dim=0)
                
                # Run batched inference
                with torch.inference_mode():
                    depth_batch, mask_batch = await run_in_threadpool(
                        processor.dap.predict, batch, da3_projection=da3_projection
                    )
                
                # Store results (move to CPU to save VRAM)
                for i in range(depth_batch.shape[0]):
                    depths.append(depth_batch[i].cpu())
                    if mask_batch is not None:
                        masks.append(mask_batch[i].cpu())
                
                job.current_frame = batch_end
                job.last_updated = time.time()
            
            # ─────────────────────────────────────────────────────────────
            # PASS 2: Temporal Smoothing + SHARP Refinement + Conversion
            # ─────────────────────────────────────────────────────────────
            manifest = {"fps": fps, "frames": []}
            prev_depth = None
            prev_ref_opacities = None
            prev_ref_scales = None
            prev_ref_colors = None

            # Calculate effective alpha (0 = no smoothing, higher = more smoothing)
            # Invert for EMA: alpha=0.7 means 70% new frame, 30% previous
            ema_alpha = 1.0 - temporal_alpha if temporal_alpha > 0 else 1.0

            # Imports for conversion
            from spag4d.spherical_grid import create_spherical_grid
            from spag4d.gaussian_converter import equirect_to_gaussians, equirect_to_gaussians_refined
            from spag4d.splat_writer import save_splat

            for i, frame_path in enumerate(frames):
                job.current_frame = i + 1
                job.last_updated = time.time()

                # Get depth (move back to GPU)
                depth = depths[i].to(processor.device)
                mask = masks[i].to(processor.device) if masks else None

                # Apply temporal smoothing to depth (EMA)
                if prev_depth is not None and temporal_alpha > 0:
                    depth = ema_alpha * depth + (1 - ema_alpha) * prev_depth
                prev_depth = depth.clone()

                # Apply global scale
                depth = depth * global_scale

                # Use mask from DAP + Sky threshold
                validity_mask = mask  # From batch prediction

                # Apply sky threshold
                if sky_threshold > 0:
                    sky_mask = (depth <= sky_threshold).float()
                    if validity_mask is not None:
                        validity_mask = validity_mask * sky_mask
                    else:
                        validity_mask = sky_mask

                # Load image for conversion
                img = Image.open(frame_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img)).to(processor.device)

                # Create spherical grid
                H, W = img_tensor.shape[:2]
                grid = create_spherical_grid(H, W, processor.device, stride=stride)

                # SHARP refinement with temporal smoothing
                refined_attrs = None
                if processor.sharp_refiner is not None:
                    try:
                        processor.sharp_refiner.load_model()  # no-op if already loaded
                        img_float = img_tensor.float() / 255.0
                        raw_attrs = processor.sharp_refiner.refine(img_float, depth)
                    except Exception as e:
                        import warnings
                        warnings.warn(f"SHARP refinement failed: {e}. Falling back to geometric-only.")
                        raw_attrs = None

                    if raw_attrs is not None:
                        # Temporally smooth SHARP attributes (opacity, scale, color)
                        ref_op = raw_attrs.opacities
                        ref_sc = raw_attrs.scales
                        ref_col = raw_attrs.colors

                        if prev_ref_opacities is not None and temporal_alpha > 0:
                            ref_op = ema_alpha * ref_op + (1 - ema_alpha) * prev_ref_opacities.to(processor.device)
                            ref_sc = ema_alpha * ref_sc + (1 - ema_alpha) * prev_ref_scales.to(processor.device)
                            if ref_col is not None and prev_ref_colors is not None:
                                ref_col = ema_alpha * ref_col + (1 - ema_alpha) * prev_ref_colors.to(processor.device)

                        prev_ref_opacities = ref_op.cpu().clone()
                        prev_ref_scales = ref_sc.cpu().clone()
                        if ref_col is not None:
                            prev_ref_colors = ref_col.cpu().clone()

                        from spag4d.sharp_refiner import RefinedAttributes
                        refined_attrs = RefinedAttributes(
                            opacities=ref_op, scales=ref_sc, colors=ref_col
                        )

                # Convert to gaussians
                if refined_attrs is not None:
                    gaussians = equirect_to_gaussians_refined(
                        image=img_tensor,
                        depth=depth,
                        grid=grid,
                        refined_attrs=refined_attrs,
                        scale_factor=scale_factor,
                        thickness_ratio=thickness,
                        depth_min=depth_min,
                        depth_max=depth_max,
                        validity_mask=validity_mask,
                        scale_blend=scale_blend,
                        opacity_blend=opacity_blend,
                    )
                else:
                    gaussians = equirect_to_gaussians(
                        image=img_tensor,
                        depth=depth,
                        grid=grid,
                        scale_factor=scale_factor,
                        thickness_ratio=thickness,
                        depth_min=depth_min,
                        depth_max=depth_max,
                        validity_mask=validity_mask,
                    )

                # Apply stabilization if enabled
                if vo is not None:
                    # Process frame for visual odometry
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    vo.process_frame(img_bgr)

                    # Get stabilization rotation and transform gaussians
                    R_stab = vo.get_stabilization_rotation()
                    from spag4d.visual_odometry import transform_gaussians
                    gaussians = transform_gaussians(gaussians, R_stab)

                # Save splat
                base_name = frame_path.stem
                out_splat = output_dir / f"{base_name}.splat"
                await run_in_threadpool(save_splat, gaussians, str(out_splat))

                manifest["frames"].append(f"{base_name}.splat")
            
            # Save manifest
            import json
            with open(output_dir / "manifest.json", "w") as f:
                json.dump(manifest, f)
            
            job.frame_manifest = manifest["frames"]
            
            # 3. Zip result
            await run_in_threadpool(
                shutil.make_archive,
                str(job.output_zip_path.with_suffix('')),
                'zip',
                output_dir
            )
            
            job.output_splat_path = output_dir
            job.status = "complete"
            
            # Cleanup
            if job.input_path.exists():
                job.input_path.unlink()
            shutil.rmtree(frames_dir, ignore_errors=True)
            
    except Exception as e:
        job.status = "error"
        job.error = str(e)
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
        response["preview_url"] = f"/api/preview/{job_id}"
        response["depth_preview_url"] = f"/api/depth_preview/{job_id}"
    
    if job.status == "error":
        response["error"] = job.error

    if job.is_video:
        response["is_video"] = True
        response["total_frames"] = job.total_frames
        response["current_frame"] = job.current_frame
        if job.status == "complete":
            response["preview_manifest_url"] = f"/api/preview_video/{job_id}/manifest.json"
            response["zip_url"] = f"/api/download_video/{job_id}"
    
    # Include params used
    if job.params:
        response["params"] = job.params

    return JSONResponse(response)


@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """Get low-res SPLAT preview for web viewer."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Job not complete")
    
    if not job.preview_splat_path or not job.preview_splat_path.exists():
        raise HTTPException(404, "Preview not available")
    
    return FileResponse(
        job.preview_splat_path,
        media_type="application/octet-stream",
        filename=f"preview_{job_id[:8]}.splat"
    )


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
async def download_file(job_id: str, format: str = "ply"):
    """Download the generated file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Job not complete")
    
    if format == "splat":
        path = job.output_splat_path
        filename = f"spag4d_{job_id[:8]}.splat"
    else:
        path = job.output_ply_path
        filename = f"spag4d_{job_id[:8]}.ply"
    
    if not path or not path.exists():
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/api/preview_video/{job_id}/{filename}")
async def get_video_frame(job_id: str, filename: str):
    """Serve individual frames or manifest for video preview."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "complete":
         raise HTTPException(400, "Job not complete")
    
    # job.output_splat_path serves as the directory for video frames
    file_path = job.output_splat_path / filename
    
    if not file_path.exists():
        raise HTTPException(404, "File not found")
        
    return FileResponse(file_path)


@app.get("/api/download_video/{job_id}")
async def download_video_zip(job_id: str):
    """Download the full video sequence as ZIP."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    
    if not job.output_zip_path.exists():
        raise HTTPException(404, "File not found")
        
    return FileResponse(
        job.output_zip_path,
        media_type="application/zip",
        filename=f"spag4d_video_{job_id[:8]}.zip"
    )


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
