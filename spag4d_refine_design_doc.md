# SPAG-4D Refine Module — Design Document v7.0

## GSFix3D-Based Disocclusion Repair for Panorama-Sourced 3DGS

**Author:** Cedar Connor  
**Date:** April 2026  
**Status:** Design — Replaces FLUX Klein Track 1 / InSpatio-WorldFM Track 2 Architecture  
**Target Hardware:** Single NVIDIA A6000 (48 GB VRAM)

---

## 1. Problem Statement

SPAG-4D currently converts 360° equirectangular panoramas into 3D Gaussian Splats via DAP depth estimation and per-pixel Gaussian unprojection. The resulting splats have a structural deficiency: **disocclusion holes** wherever the original panorama's single-viewpoint capture could not observe (behind foreground objects, under furniture, around depth discontinuities).

The previous `spag4d_refine` design (v6.1) attempted to solve this with a two-track approach:

- **Track 1:** FLUX.2 Klein 9B with ml-sharp LoRA — pinhole-camera renders of hole regions fed to FLUX for 2D inpainting, then back-projected into 3D using estimated depth.
- **Track 2:** InSpatio-WorldFM — escalation target for structurally complex hallucinations that Track 1 could not resolve.

**Why this failed:** The depth alignment problem is structural, not tunable. FLUX generates plausible 2D pixels but has no awareness of 3D scene geometry. Back-projecting those pixels requires accurate depth for every inpainted pixel — and there is no ground truth depth for content that was never observed. The result: angle-specific generations look reasonable in isolation, but Gaussians land at wrong depths, producing parallax errors, floating splats, and seam artifacts when viewed from any other angle.

**The fix:** Replace the "generate in 2D, back-project to 3D" paradigm with "render holes from 3DGS, repair the renders with a scene-adapted diffusion model, optimize the 3DGS to match the repaired renders." Depth placement becomes an optimization variable solved by differentiable rendering, not an input you must get right up-front.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EXISTING SPAG-4D PIPELINE                │
│                                                             │
│  Panorama ─→ DAP Depth ─→ Equirect Unproject ─→ Raw PLY    │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  NEW: spag4d_refine MODULE                   │
│                                                             │
│  ┌───────────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │ Phase 1       │   │ Phase 2      │   │ Phase 3       │  │
│  │ Camera Rig &  │──▶│ GSFixer      │──▶│ 3DGS Distill  │  │
│  │ Hole Detection│   │ Fine-tune +  │   │ (refine_gs)   │  │
│  │               │   │ Inference    │   │               │  │
│  └───────────────┘   └──────────────┘   └───────────────┘  │
│         │                    │                   │           │
│    Novel-view           Repaired            Refined PLY     │
│    renders +            images              (holes filled)  │
│    hole masks                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Dependency

**GSFix3D** (github.com/GSFix3D/GSFix3D)  
- Paper: arXiv 2508.14717, accepted 3DV 2026  
- License: Apache 2.0 + Gaussian-Splatting-License (non-commercial)  
- Backbone: Stable Diffusion v2 (fits A6000 comfortably)  
- Pretrained checkpoints: huggingface.co/collections/goldoak1421/gsfix3d  
- Tested on: Ubuntu 22.04, Python 3.11, CUDA 12.1  

---

## 3. Why GSFix3D Over Alternatives

| Criterion | FLUX Klein (current) | GSFix3D | GSFixer (GVCLab) | RI3D |
|-----------|---------------------|---------|-------------------|------|
| Code available | N/A (custom) | ✅ Nov 2025 | ✅ Released | ❌ "Coming soon" |
| Pretrained models | N/A | ✅ HuggingFace | ✅ HuggingFace | ❌ |
| VRAM requirement | ~24 GB (Klein 9B) | ~16-20 GB (SD v2) | 40+ GB (CogVideoX DiT) | Unknown |
| Fits A6000? | Yes (tight) | Yes (comfortable) | Marginal | Unknown |
| Handles inpainting? | 2D only | ✅ Random mask augmentation | Artifact repair only | ✅ Dedicated model |
| Depth alignment | Manual back-projection (broken) | Automatic via 3DGS optimization | Automatic via 3DGS optimization | Automatic via 3DGS optimization |
| Multi-view consistency | None (per-frame) | Via iterative distillation | Via video diffusion prior | Via two-stage optimization |
| Dual conditioning (mesh+GS) | No | ✅ | No | No |
| Scene-specific fine-tuning | LoRA (ml-sharp) | ✅ Minutes per scene | Yes | Yes |
| 360° panorama tested | No | No (adaptation needed) | No | No |

**Decision:** GSFix3D provides the best balance of capability, VRAM budget, and code maturity. The dual-conditioning (mesh + 3DGS renders) is uniquely valuable for panorama-sourced splats where you can extract a rough mesh from the DAP depth to give the diffusion model structural context.

---

## 4. Detailed Phase Design

### Phase 1: Camera Rig Generation & Hole Detection

**Goal:** Produce a set of novel-view renders from the raw 3DGS that expose disocclusion holes, along with binary masks identifying hole regions.

#### 4.1.1 Camera Placement Strategy

Panorama-sourced splats have dense angular coverage from a single origin but zero translational baseline. Disocclusion holes only become visible when the camera **translates** away from the capture origin.

```python
# Camera rig generation for panorama-sourced splats
#
# Strategy: Translate inward toward scene content along multiple
# directions, then render perspective views. Holes appear at
# depth discontinuities when parallax reveals unseen surfaces.

def generate_camera_rig(
    origin: np.ndarray,          # [0, 0, 0] — panorama capture position
    depth_map: np.ndarray,       # DAP depth (equirectangular)
    num_directions: int = 12,    # Azimuthal directions to sample
    num_depths: int = 3,         # Translation distances per direction
    fov_deg: float = 60.0,       # Perspective camera FOV
    translation_fracs: tuple = (0.05, 0.15, 0.30),  # Fraction of median depth
) -> list[CameraPose]:
    """
    For each azimuthal direction:
      - Compute median scene depth in that direction
      - Place cameras at translation_fracs * median_depth along that ray
      - Orient camera to look back toward scene content
      - This maximizes parallax at depth discontinuities
    """
    cameras = []
    median_depth = np.median(depth_map[depth_map > 0])

    for azi_idx in range(num_directions):
        azimuth = (2 * np.pi * azi_idx) / num_directions

        for frac in translation_fracs:
            t = frac * median_depth
            cam_pos = origin + t * np.array([
                np.cos(azimuth), 0.0, np.sin(azimuth)
            ])

            # Look toward scene center (back toward origin)
            look_at = origin
            cam = perspective_camera(
                position=cam_pos,
                look_at=look_at,
                fov=fov_deg,
                resolution=(512, 512)  # GSFixer operates at 512x512
            )
            cameras.append(cam)

    return cameras  # 12 directions × 3 depths = 36 cameras
```

#### 4.1.2 Rendering & Hole Mask Extraction

Render each camera from the raw 3DGS. Holes manifest as regions with zero or near-zero alpha accumulation.

```python
def render_with_hole_mask(
    gaussians: GaussianModel,
    camera: CameraPose,
    alpha_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        rgb: (H, W, 3) rendered image
        hole_mask: (H, W) binary mask, 1 = hole (alpha < threshold)
    """
    render_result = rasterize(gaussians, camera)
    rgb = render_result.rgb            # (H, W, 3)
    alpha = render_result.alpha        # (H, W, 1)

    hole_mask = (alpha.squeeze() < alpha_threshold).float()
    return rgb, hole_mask
```

#### 4.1.3 Camera Selection / Prioritization

Not all 36 cameras will have significant holes. Filter to those with >5% hole pixels to avoid wasting fine-tuning and inference budget on views that don't need repair.

```python
def select_repair_cameras(
    cameras: list[CameraPose],
    hole_masks: list[np.ndarray],
    min_hole_fraction: float = 0.05,
    max_cameras: int = 20,
) -> list[int]:
    """Select cameras with significant holes, up to max_cameras."""
    scored = []
    for i, mask in enumerate(hole_masks):
        frac = mask.mean()
        if frac >= min_hole_fraction:
            scored.append((i, frac))

    # Sort by hole fraction descending, take top-N
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:max_cameras]]
```

---

### Phase 2: GSFixer Fine-Tuning & Inference

**Goal:** Adapt the pretrained GSFixer diffusion model to this specific scene, then generate repaired versions of the hole-containing renders.

#### 4.2.1 Scene-Specific Fine-Tuning

GSFix3D's fine-tuning protocol adapts Stable Diffusion v2 to the current scene using the existing (non-hole) renders as training data. This teaches the model the scene's appearance, materials, and lighting so it can plausibly extend those into hole regions.

**Training data preparation:**

```python
def prepare_finetune_data(
    gaussians: GaussianModel,
    panorama: np.ndarray,        # Original equirectangular image
    depth_map: np.ndarray,       # DAP depth
) -> dict:
    """
    Generate clean perspective renders from the ORIGINAL capture viewpoint
    (where we have ground truth) for fine-tuning GSFixer.

    These are cubemap face extractions — 6 perspective views from origin
    where the 3DGS should perfectly match the panorama.
    """
    # Extract 6 cubemap faces from equirectangular
    cubemap_faces = equirect_to_cubemap(panorama, face_size=512)

    # Render same views from 3DGS
    cubemap_cameras = get_cubemap_cameras(origin=[0,0,0], face_size=512)
    gs_renders = [rasterize(gaussians, cam).rgb for cam in cubemap_cameras]

    # Training pairs: gs_render (input) → cubemap_face (target)
    # This teaches GSFixer what "good" looks like for this scene
    pairs = []
    for gs_img, gt_img in zip(gs_renders, cubemap_faces):
        pairs.append({
            'input': gs_img,       # What the GS currently renders
            'target': gt_img,      # What it should look like
            'condition_gs': gs_img,
        })

    return pairs
```

**Fine-tuning configuration (adapted from GSFix3D defaults):**

```yaml
# config/spag4d_finetune.yaml
base_model: "pretrained/gsfix3d_base.ckpt"   # From HuggingFace
learning_rate: 1.0e-5
train_steps: 500          # ~5 minutes on A6000
batch_size: 1
resolution: 512
use_mesh_conditioning: true   # Dual conditioning: mesh + GS renders
use_gs_conditioning: true
random_mask_augmentation: true  # Critical: teaches inpainting
mask_ratio_range: [0.1, 0.5]   # Random hole sizes during training
```

**Mesh extraction for dual conditioning:**

```python
def extract_conditioning_mesh(
    depth_map: np.ndarray,
    panorama: np.ndarray,
    simplify_ratio: float = 0.1,
) -> trimesh.Trimesh:
    """
    Extract a rough textured mesh from DAP depth for GSFixer's
    mesh conditioning branch.

    This gives the diffusion model geometric context that pure
    GS renders lack — especially around depth discontinuities
    where holes form.
    """
    # Unproject depth to point cloud
    points, colors = equirect_depth_to_pointcloud(depth_map, panorama)

    # Create mesh via Poisson reconstruction or simple triangulation
    mesh = poisson_surface_reconstruction(points, colors)

    # Simplify for conditioning (don't need full detail)
    mesh = mesh.simplify_quadric_decimation(
        int(len(mesh.faces) * simplify_ratio)
    )

    return mesh
```

#### 4.2.2 Inference: Generating Repaired Views

After fine-tuning, run GSFixer on all selected hole-containing renders.

```python
def run_gsfixer_inference(
    model: GSFixerModel,
    gs_renders: list[np.ndarray],        # Renders with holes
    hole_masks: list[np.ndarray],        # Binary hole masks
    mesh: trimesh.Trimesh,               # Conditioning mesh
    cameras: list[CameraPose],           # Camera poses for mesh rendering
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> list[np.ndarray]:
    """
    For each hole-containing render:
    1. Render mesh from same camera → mesh_condition
    2. Compose input: gs_render with holes visible
    3. Run GSFixer reverse diffusion
    4. Output: clean image with holes filled
    """
    repaired = []

    for gs_img, mask, cam in zip(gs_renders, hole_masks, cameras):
        # Render mesh from this camera for dual conditioning
        mesh_render = render_mesh(mesh, cam, resolution=(512, 512))

        # GSFixer inference
        repaired_img = model.sample(
            gs_image=gs_img,
            mesh_image=mesh_render,
            mask=mask,             # Tells model where to focus inpainting
            num_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        repaired.append(repaired_img)

    return repaired
```

#### 4.2.3 Iterative Inpainting for Large Holes

For panorama-sourced splats, some disocclusion regions can be very large (entire walls behind furniture). A single pass may not produce coherent fills. Following GSFix3D's protocol:

```python
def iterative_repair(
    model: GSFixerModel,
    gaussians: GaussianModel,
    cameras: list[CameraPose],
    mesh: trimesh.Trimesh,
    max_iterations: int = 3,
    convergence_threshold: float = 0.02,  # Fraction of remaining holes
) -> GaussianModel:
    """
    Iterative repair loop:
    1. Render → detect holes → repair → distill back to 3DGS
    2. Re-render → detect remaining holes → repair → distill
    3. Repeat until holes < threshold or max iterations
    """
    for iteration in range(max_iterations):
        # Render and detect holes
        renders, masks = [], []
        for cam in cameras:
            rgb, mask = render_with_hole_mask(gaussians, cam)
            renders.append(rgb)
            masks.append(mask)

        # Check convergence
        avg_hole_frac = np.mean([m.mean() for m in masks])
        print(f"Iteration {iteration}: avg hole fraction = {avg_hole_frac:.4f}")
        if avg_hole_frac < convergence_threshold:
            break

        # Select views needing repair
        repair_indices = select_repair_cameras(
            cameras, masks, min_hole_fraction=0.02
        )

        if len(repair_indices) == 0:
            break

        # Run GSFixer on selected views
        repair_renders = [renders[i] for i in repair_indices]
        repair_masks = [masks[i] for i in repair_indices]
        repair_cams = [cameras[i] for i in repair_indices]

        repaired = run_gsfixer_inference(
            model, repair_renders, repair_masks, mesh, repair_cams
        )

        # Distill repaired images back into 3DGS (Phase 3)
        gaussians = distill_to_gaussians(
            gaussians, repaired, repair_cams, repair_masks
        )

    return gaussians
```

---

### Phase 3: 3DGS Distillation

**Goal:** Optimize the 3DGS representation so that rendering from the repaired viewpoints matches the GSFixer outputs. This is where depth alignment happens automatically.

#### 4.3.1 Optimization Loop

This adapts GSFix3D's `refine_gs.py` script for the SPAG-4D context.

```python
def distill_to_gaussians(
    gaussians: GaussianModel,
    repaired_images: list[np.ndarray],    # Pseudo-GT from GSFixer
    cameras: list[CameraPose],
    hole_masks: list[np.ndarray],
    original_images: list[np.ndarray] = None,  # Optional: cubemap GTs
    original_cameras: list[CameraPose] = None,
    num_iterations: int = 3000,
    densify_interval: int = 100,
    densify_grad_threshold: float = 0.0002,
    lr_position: float = 0.00016,
    lr_feature: float = 0.0025,
    lr_opacity: float = 0.05,
    lr_scaling: float = 0.005,
    lr_rotation: float = 0.001,
) -> GaussianModel:
    """
    Key insight: The loss is computed in IMAGE SPACE against the
    repaired renders. The optimizer adjusts Gaussian positions,
    scales, rotations, and colors to minimize this loss.

    Depth alignment is an emergent property — the optimizer places
    new Gaussians at whatever 3D positions produce the correct
    2D appearance across multiple viewpoints simultaneously.
    """
    optimizer = setup_optimizer(gaussians, lr_position, lr_feature,
                                lr_opacity, lr_scaling, lr_rotation)

    for step in range(num_iterations):
        # Sample a training view
        if original_images and random.random() < 0.3:
            # 30% of time: train on original (non-hole) views
            # to prevent drift in already-good regions
            idx = random.randint(0, len(original_images) - 1)
            target = original_images[idx]
            camera = original_cameras[idx]
            mask_weight = None
        else:
            # 70% of time: train on repaired views
            idx = random.randint(0, len(repaired_images) - 1)
            target = repaired_images[idx]
            camera = cameras[idx]
            # Weight loss higher in hole regions to focus optimization
            mask_weight = hole_masks[idx] * 2.0 + 1.0

        # Forward render
        render = rasterize(gaussians, camera)

        # Loss: L1 + D-SSIM (standard 3DGS loss)
        l1_loss = l1(render.rgb, target)
        ssim_loss = 1.0 - ssim(render.rgb, target)
        loss = 0.8 * l1_loss + 0.2 * ssim_loss

        # Apply mask weighting if available
        if mask_weight is not None:
            loss = (loss * mask_weight).mean()
        else:
            loss = loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Adaptive density control
        if step % densify_interval == 0 and step < num_iterations * 0.8:
            gaussians.densify_and_prune(
                grad_threshold=densify_grad_threshold,
                min_opacity=0.005,
                max_screen_size=20.0,
            )

    return gaussians
```

#### 4.3.2 Preserving Original Content

Critical constraint: the refinement must not degrade regions that were already well-reconstructed from the panorama. Two mechanisms:

1. **Mixed training schedule** (shown above): 30% of optimization steps use the original cubemap ground truth views, anchoring already-good Gaussians.

2. **Gaussian freezing by provenance:**

```python
def tag_gaussian_provenance(gaussians, hole_masks, cameras):
    """
    Mark which Gaussians are 'original' (from panorama depth)
    vs. 'new' (created during refinement densification).

    Original Gaussians get lower learning rates to prevent drift.
    """
    # Render each camera, accumulate per-Gaussian gradient
    # Gaussians only visible through hole regions are 'refinement' Gaussians
    # Gaussians visible in non-hole regions are 'original'
    for g_idx in range(len(gaussians)):
        if gaussians.provenance[g_idx] == 'original':
            # Reduce LR by 10x for original Gaussians
            gaussians.lr_scale[g_idx] = 0.1
        else:
            gaussians.lr_scale[g_idx] = 1.0
```

---

## 5. File Structure

```
SPAG-4D/
├── spag4d/
│   ├── core.py                    # Existing orchestrator (unchanged)
│   ├── cli.py                     # Add --refine flag
│   ├── dap_arch/                  # Existing DAP wrapper
│   │
│   └── refine/                    # NEW MODULE
│       ├── __init__.py
│       ├── pipeline.py            # Top-level refine() entry point
│       ├── camera_rig.py          # Phase 1: camera placement & hole detection
│       ├── mesh_extract.py        # Mesh extraction for dual conditioning
│       ├── gsfixer_adapter.py     # Phase 2: GSFixer fine-tune & inference wrapper
│       ├── distill.py             # Phase 3: 3DGS optimization loop
│       ├── provenance.py          # Gaussian tagging (original vs. new)
│       └── config.py              # Refinement hyperparameters
│
├── third_party/
│   └── GSFix3D/                   # Git submodule
│       ├── src/                   # GSFixer model code
│       ├── gs/                    # Gaussian splatting utilities
│       ├── scripts/               # Fine-tune & refine scripts
│       ├── config/                # Training configs
│       └── diff-gaussian-rasterization/  # Submodule
│
├── pretrained/
│   └── gsfix3d/                   # Downloaded from HuggingFace
│       ├── gsfix3d_base.ckpt      # Base pretrained weights
│       └── gsfix3d_scannet.ckpt   # ScanNet++ variant (optional)
│
└── configs/
    └── refine_defaults.yaml       # Default refinement parameters
```

---

## 6. Integration with Existing CLI

```python
# In spag4d/cli.py — add refine option to convert command

@click.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('--refine', is_flag=True, default=False,
              help='Run GSFix3D-based disocclusion repair after initial conversion')
@click.option('--refine-iterations', default=3,
              help='Max refinement iterations')
@click.option('--refine-cameras', default=36,
              help='Number of novel-view cameras for repair')
def convert(input_path, output_path, refine, refine_iterations, refine_cameras):
    converter = SPAG4D(device='cuda')
    result = converter.convert(input_path, output_path)

    if refine:
        from spag4d.refine.pipeline import refine_splat
        result = refine_splat(
            ply_path=output_path,
            panorama_path=input_path,
            depth_map=result.depth_map,
            max_iterations=refine_iterations,
            num_cameras=refine_cameras,
            output_path=output_path.replace('.ply', '_refined.ply'),
        )
```

---

## 7. Top-Level Pipeline Entry Point

```python
# spag4d/refine/pipeline.py

import torch
import numpy as np
from pathlib import Path

from .camera_rig import generate_camera_rig, render_with_hole_mask, select_repair_cameras
from .mesh_extract import extract_conditioning_mesh
from .gsfixer_adapter import load_gsfixer, finetune_gsfixer, run_gsfixer_inference
from .distill import distill_to_gaussians
from .config import RefineConfig


def refine_splat(
    ply_path: str,
    panorama_path: str,
    depth_map: np.ndarray,
    max_iterations: int = 3,
    num_cameras: int = 36,
    output_path: str = None,
    config: RefineConfig = None,
) -> dict:
    """
    Full refinement pipeline.

    Args:
        ply_path: Path to raw PLY from SPAG-4D core pipeline
        panorama_path: Original equirectangular image
        depth_map: DAP depth output (equirectangular)
        max_iterations: Refinement iterations
        num_cameras: Novel-view cameras to generate
        output_path: Where to save refined PLY
        config: Override default hyperparameters

    Returns:
        dict with refined_ply_path, stats (holes_before, holes_after, etc.)
    """
    config = config or RefineConfig()
    device = torch.device('cuda')

    # --- Load inputs ---
    panorama = load_image(panorama_path)
    gaussians = load_gaussians_from_ply(ply_path, device)

    # --- Phase 1: Camera rig & hole detection ---
    print("[refine] Phase 1: Generating camera rig...")
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth_map,
        num_directions=num_cameras // 3,
        num_depths=3,
        fov_deg=config.camera_fov,
        translation_fracs=config.translation_fracs,
    )

    # --- Extract mesh for dual conditioning ---
    print("[refine] Extracting conditioning mesh...")
    mesh = extract_conditioning_mesh(
        depth_map=depth_map,
        panorama=panorama,
        simplify_ratio=config.mesh_simplify_ratio,
    )

    # --- Prepare original views for anchoring ---
    print("[refine] Preparing anchor views from panorama...")
    cubemap_faces, cubemap_cameras = extract_cubemap_views(
        panorama, depth_map, face_size=512
    )

    # --- Phase 2: Load & fine-tune GSFixer ---
    print("[refine] Phase 2: Loading GSFixer...")
    model = load_gsfixer(
        checkpoint_path=config.gsfixer_checkpoint,
        device=device,
    )

    print("[refine] Fine-tuning GSFixer on scene...")
    finetune_gsfixer(
        model=model,
        gaussians=gaussians,
        cubemap_faces=cubemap_faces,
        cubemap_cameras=cubemap_cameras,
        mesh=mesh,
        train_steps=config.finetune_steps,
        learning_rate=config.finetune_lr,
    )

    # --- Iterative repair loop ---
    initial_hole_frac = None

    for iteration in range(max_iterations):
        print(f"\n[refine] === Iteration {iteration + 1}/{max_iterations} ===")

        # Render all cameras, detect holes
        renders, masks = [], []
        for cam in cameras:
            rgb, mask = render_with_hole_mask(
                gaussians, cam,
                alpha_threshold=config.alpha_threshold,
            )
            renders.append(rgb)
            masks.append(mask)

        avg_hole_frac = np.mean([m.mean() for m in masks])
        if initial_hole_frac is None:
            initial_hole_frac = avg_hole_frac
        print(f"[refine] Average hole fraction: {avg_hole_frac:.4f}")

        if avg_hole_frac < config.convergence_threshold:
            print("[refine] Converged — holes below threshold.")
            break

        # Select views needing repair
        repair_indices = select_repair_cameras(
            cameras, masks,
            min_hole_fraction=config.min_hole_fraction,
            max_cameras=config.max_repair_cameras,
        )
        print(f"[refine] Repairing {len(repair_indices)} views")

        if len(repair_indices) == 0:
            print("[refine] No views need repair — done.")
            break

        # GSFixer inference
        repair_renders = [renders[i] for i in repair_indices]
        repair_masks = [masks[i] for i in repair_indices]
        repair_cams = [cameras[i] for i in repair_indices]

        repaired = run_gsfixer_inference(
            model=model,
            gs_renders=repair_renders,
            hole_masks=repair_masks,
            mesh=mesh,
            cameras=repair_cams,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
        )

        # Phase 3: Distill back into 3DGS
        print(f"[refine] Distilling {len(repaired)} repaired views into 3DGS...")
        gaussians = distill_to_gaussians(
            gaussians=gaussians,
            repaired_images=repaired,
            cameras=repair_cams,
            hole_masks=repair_masks,
            original_images=cubemap_faces,
            original_cameras=cubemap_cameras,
            num_iterations=config.distill_iterations,
            densify_grad_threshold=config.densify_grad_threshold,
        )

    # --- Export refined PLY ---
    output_path = output_path or ply_path.replace('.ply', '_refined.ply')
    save_gaussians_to_ply(gaussians, output_path)

    final_hole_frac = avg_hole_frac
    print(f"\n[refine] Done. Holes: {initial_hole_frac:.4f} → {final_hole_frac:.4f}")
    print(f"[refine] Saved to: {output_path}")

    return {
        'refined_ply_path': output_path,
        'initial_hole_fraction': initial_hole_frac,
        'final_hole_fraction': final_hole_frac,
        'gaussians_count': len(gaussians),
        'iterations_used': iteration + 1,
    }
```

---

## 8. Configuration Defaults

```python
# spag4d/refine/config.py

from dataclasses import dataclass


@dataclass
class RefineConfig:
    # --- Paths ---
    gsfixer_checkpoint: str = "pretrained/gsfix3d/gsfix3d_base.ckpt"

    # --- Phase 1: Camera Rig ---
    camera_fov: float = 60.0
    translation_fracs: tuple = (0.05, 0.15, 0.30)
    alpha_threshold: float = 0.1

    # --- Phase 1: Camera Selection ---
    min_hole_fraction: float = 0.03
    max_repair_cameras: int = 20

    # --- Phase 2: GSFixer ---
    finetune_steps: int = 500
    finetune_lr: float = 1.0e-5
    inference_steps: int = 50
    guidance_scale: float = 7.5
    mesh_simplify_ratio: float = 0.1

    # --- Phase 3: Distillation ---
    distill_iterations: int = 3000
    densify_grad_threshold: float = 0.0002
    lr_position: float = 0.00016
    lr_feature: float = 0.0025
    lr_opacity: float = 0.05
    lr_scaling: float = 0.005
    lr_rotation: float = 0.001
    original_view_ratio: float = 0.3     # Fraction of steps using GT views

    # --- Convergence ---
    convergence_threshold: float = 0.02  # Stop when avg holes < 2%
    max_iterations: int = 3
```

---

## 9. Adaptation Challenges & Mitigations

### 9.1 Panorama vs. Multi-View Input Distribution

GSFix3D was trained/tested on SLAM reconstructions (Replica, ScanNet++) with multi-camera trajectories. SPAG-4D's input is fundamentally different — one viewpoint, 360° coverage.

**Mitigation:** The fine-tuning step adapts GSFixer to the current scene's appearance. The key adaptation is in camera rig design (Section 4.1): we explicitly generate the translational diversity that SLAM data naturally provides, by displacing cameras away from the capture origin.

### 9.2 Gaussian Splatting Format Compatibility

GSFix3D uses the original `diff-gaussian-rasterization` from Inria. SPAG-4D currently writes PLY in standard 3DGS format. The formats should be compatible, but verify:

- SH coefficient ordering (standard 3DGS order)
- Scale representation (log-scale vs. absolute)
- Rotation convention (quaternion wxyz vs. xyzw)
- Opacity representation (logit-sigmoid vs. raw)

**Action item:** Write a conversion utility in `spag4d/refine/format_compat.py` that handles any discrepancies between SPAG-4D's PLY output and GSFix3D's expected input format.

### 9.3 Mesh Conditioning Quality

GSFix3D's dual conditioning expects reasonable mesh quality. DAP depth from a single panorama will produce a rough mesh with holes and noise at depth discontinuities — exactly where we need the most help.

**Mitigation:** Use aggressive mesh simplification (10% of faces) and hole-filling before passing to GSFixer. The mesh is a conditioning signal, not a reconstruction target — it just needs to convey approximate scene structure. Even a rough mesh dramatically outperforms GS-only conditioning for large disocclusion regions.

### 9.4 VRAM Budget

Per-phase peak VRAM estimates on A6000 (48 GB):

| Phase | Component | Est. VRAM |
|-------|-----------|-----------|
| Phase 1 | 3DGS render (diff-gaussian-rasterization) | ~4-8 GB |
| Phase 2 | GSFixer fine-tune (SD v2 + gradients) | ~20-24 GB |
| Phase 2 | GSFixer inference (SD v2, no gradients) | ~12-16 GB |
| Phase 3 | 3DGS optimization (Gaussians + gradients) | ~8-16 GB |

Phases are sequential, not concurrent. Peak is Phase 2 fine-tuning at ~24 GB — well within A6000 headroom. If VRAM is tight, fine-tune with gradient checkpointing or reduce batch resolution to 384×384.

---

## 10. Installation Steps

```bash
# From SPAG-4D root directory

# 1. Add GSFix3D as submodule
git submodule add https://github.com/GSFix3D/GSFix3D.git third_party/GSFix3D
git submodule update --init --recursive

# 2. Install GSFix3D dependencies (into existing SPAG-4D venv)
pip install -r third_party/GSFix3D/requirements.txt

# 3. Build diff-gaussian-rasterization
cd third_party/GSFix3D/diff-gaussian-rasterization
pip install .
cd ../../..

# 4. Download pretrained GSFixer checkpoint
mkdir -p pretrained/gsfix3d
# Download from https://huggingface.co/collections/goldoak1421/gsfix3d
# Place checkpoint(s) in pretrained/gsfix3d/

# 5. Install additional SPAG-4D refine dependencies
pip install trimesh open3d  # For mesh extraction
```

---

## 11. Testing Strategy

### Unit Tests

```
tests/
├── test_camera_rig.py        # Camera placement geometry correctness
├── test_hole_detection.py    # Synthetic splat with known holes
├── test_mesh_extract.py      # Mesh from synthetic depth
├── test_format_compat.py     # PLY round-trip consistency
└── test_distill.py           # Optimization reduces loss on synthetic target
```

### Integration Test

```python
def test_full_pipeline_synthetic():
    """
    Create a synthetic panorama with known depth,
    generate raw splat, run refinement, verify:
    1. Hole fraction decreases
    2. Original-view PSNR does not degrade by > 0.5 dB
    3. Novel-view PSNR improves by > 2 dB
    """
    pass
```

### Qualitative Evaluation

Use the existing SPAG-4D web viewer with A/B comparison:
- Raw splat (before refinement) vs. refined splat
- Fly-through paths that traverse disocclusion regions
- Measure artifacts: floaters, seams, color inconsistency

---

## 12. Migration Path from FLUX Klein

The transition is clean because FLUX Klein was never implemented in the codebase — it existed only as a design spec (v6.1). No code needs to be removed or deprecated. The `spag4d/refine/` module is purely additive.

**What's preserved from v6.1 design thinking:**
- Forward-warp disocclusion prior → now used to inform camera rig placement (Section 4.1)
- Two-pass escalation heuristic → replaced by iterative convergence loop (Section 4.2.3)
- Provenance tagging → carried forward (Section 4.3.2)

**What's abandoned:**
- FLUX.2 Klein 9B dependency
- ml-sharp LoRA
- InSpatio-WorldFM as Track 2 backend
- Manual depth alignment / back-projection from 2D inpaints
- Pinhole-camera-based refinement architecture

---

## 13. Future Considerations

### 13.1 Video Diffusion Upgrade Path

If GSFixer (GVCLab) — the DiT/CogVideoX-based approach — becomes viable on A6000 (via quantization or architectural improvements), it would be a natural upgrade. Video diffusion provides multi-view consistency natively rather than through iterative distillation. The Phase 1 (camera rig) and Phase 3 (distillation) modules would remain unchanged; only Phase 2 swaps.

### 13.2 RI3D Integration (When Code Ships)

RI3D's two-model architecture (dedicated repair + dedicated inpainting) may outperform GSFix3D's unified approach for panorama-sourced splats where the distinction between "artifact repair" and "missing content hallucination" is very clear. Monitor the repo for code release.

### 13.3 DreamScene360 as Upstream Generator

DreamScene360/ComfyUI-DreamScene360 remains a viable upstream text-to-panorama generator feeding into SPAG-4D. This refinement module operates downstream and is agnostic to panorama source.

### 13.4 Depth Estimator Upgrades

DAP produces the initial depth, but the refinement loop's distillation step implicitly corrects depth errors (by optimizing Gaussian positions). Future work: compare DAP vs. DA360 depth as initialization — the refinement loop may be robust enough that initial depth quality matters less than before.
