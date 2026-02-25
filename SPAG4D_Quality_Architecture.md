# SPAG-4D Quality Improvement Architecture
## Design Document v1.0

**Project:** SPAG-4D — 360° Panorama to Gaussian Splat  
**Author:** Cedar / Claude  
**Date:** February 2026  
**Scope:** Six-phase quality improvement plan covering depth fusion, Gaussian parameterization, artifact reduction, and post-processing.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Baseline](#2-current-architecture-baseline)
3. [Phase 1: Cubemap Depth Pro + DAP Scale Alignment](#3-phase-1-cubemap-depth-pro--dap-scale-alignment)
4. [Phase 2: Depth-Aware Gaussian Sizing + Normal-Oriented Covariances](#4-phase-2-depth-aware-gaussian-sizing--normal-oriented-covariances)
5. [Phase 3: Sky Detection & Pole Thinning](#5-phase-3-sky-detection--pole-thinning)
6. [Phase 4: Feathered Blending / Laplacian Fusion](#6-phase-4-feathered-blending--laplacian-fusion)
7. [Phase 5: Adaptive Stride](#7-phase-5-adaptive-stride)
8. [Phase 6: Tangent Patch Projection (Upgrade Path)](#8-phase-6-tangent-patch-projection-upgrade-path)
9. [New Module Map](#9-new-module-map)
10. [CLI / API Surface Changes](#10-cli--api-surface-changes)
11. [VRAM Budget & Performance](#11-vram-budget--performance)
12. [Testing Strategy](#12-testing-strategy)
13. [References](#13-references)

---

## 1. Executive Summary

SPAG-4D currently uses DAP (Depth Any Panoramas) for equirectangular-aware depth estimation, then unprojects pixels to 3D Gaussians with uniform parameters. The output quality is limited by three categories of problems:

**Depth quality:** DAP provides globally consistent but low-frequency depth. Fine edges and object boundaries are mushy. SHARP integration helps but operates on the same underlying depth model.

**Gaussian parameterization:** Uniform stride, uniform scale, isotropic covariance, and no confidence filtering produce splats that are too large in the foreground, too small in the background, and blobby everywhere.

**Structural artifacts:** Sky regions produce garbage geometry, pole convergence creates density spikes, and cubemap seams (if using perspective depth models) produce visible discontinuities.

This document specifies six phases of improvements, ordered by impact-to-effort ratio. Each phase is independently deployable and backward-compatible with the existing CLI/API.

---

## 2. Current Architecture Baseline

```
┌─────────────────────────────────────────────────────┐
│                    CURRENT PIPELINE                  │
│                                                      │
│  ERP Image ──► DAP Depth ──► (Optional SHARP) ──►   │
│  Uniform Unproject ──► Isotropic Gaussians ──► PLY   │
└─────────────────────────────────────────────────────┘
```

### Current `core.py` flow (inferred from README + API surface):

1. Load equirectangular image (H × W, typically 2:1 ratio)
2. Run DAP model → depth map (H × W, metric depth in meters)
3. (Optional) Run SHARP refiner → refined depth with edge enhancement
4. For each pixel at stride interval:
   - Convert (row, col) → spherical (θ, φ)
   - Unproject: `xyz = depth * [sin(θ)cos(φ), cos(θ), sin(θ)sin(φ)]`
   - Sample RGB from input image
   - Create Gaussian with uniform `scale_factor` and isotropic covariance
5. Write PLY / SPLAT

### Current parameters:

| Parameter | Default | Role |
|-----------|---------|------|
| `stride` | 2 | Uniform pixel skip |
| `scale_factor` | 1.5 | Global Gaussian radius multiplier |
| `thickness` | 0.1 | Radial thickness ratio |
| `global_scale` | 1.0 | Depth scale correction |
| `depth_min` | 0.1 | Minimum clamp (meters) |
| `depth_max` | 100.0 | Maximum clamp (meters) |

---

## 3. Phase 1: Cubemap Depth Pro + DAP Scale Alignment

### 3.1 Objective

Combine DAP's equirectangular-aware global depth structure with Apple Depth Pro's superior boundary sharpness on perspective images. DAP provides the "skeleton" (correct global geometry, no seam issues), and Depth Pro provides the "skin" (sharp edges, fine detail).

### 3.2 Why Not Run Depth Pro on the Full Equirect?

Depth Pro was trained exclusively on perspective images. Feeding it an equirectangular image produces:
- Severe pole distortion artifacts (it interprets stretching as real geometry)
- Incorrect metric scale (its focal length estimator assumes perspective projection)
- No awareness of wrap-around continuity at the left/right border

Running DAP on the full equirect first is essential for global consistency.

### 3.3 Projection Strategy: Cubemap with Extended FOV

Project the equirectangular image onto 6 cubemap faces, each with an extended field of view to create overlap zones for blending.

```
Standard cubemap face:  90° × 90° FOV
Extended cubemap face: 100° × 100° FOV (10° overlap on each edge)
```

#### 3.3.1 Equirect → Cubemap Projection

For each cubemap face direction `d` ∈ {+X, -X, +Y, -Y, +Z, -Z}:

```python
def equirect_to_cubemap_face(erp_image, face_dir, face_size=1024, fov_deg=100):
    """
    Project equirectangular image to a single cubemap face.
    
    Args:
        erp_image: (H, W, 3) equirectangular image
        face_dir: one of 'front', 'back', 'left', 'right', 'up', 'down'
        face_size: output resolution per face
        fov_deg: field of view in degrees (>90 for overlap)
    
    Returns:
        face_rgb: (face_size, face_size, 3)
        face_uv_map: (face_size, face_size, 2) mapping back to ERP coords
    """
    # Build pixel grid for the face
    f = face_size / (2 * np.tan(np.radians(fov_deg / 2)))
    u = np.linspace(-face_size/2, face_size/2, face_size)
    v = np.linspace(-face_size/2, face_size/2, face_size)
    uu, vv = np.meshgrid(u, v)
    
    # Ray directions in face-local coords
    # (face_dir determines rotation matrix R)
    R = get_face_rotation(face_dir)  # 3x3 rotation
    rays = np.stack([uu/f, vv/f, np.ones_like(uu)], axis=-1)
    rays = rays @ R.T
    
    # Convert rays to spherical → ERP pixel coords
    theta = np.arccos(rays[..., 1] / np.linalg.norm(rays, axis=-1))
    phi = np.arctan2(rays[..., 0], rays[..., 2])
    
    erp_x = (phi / np.pi + 1) / 2 * W
    erp_y = theta / np.pi * H
    
    # Bilinear sample
    face_rgb = bilinear_sample(erp_image, erp_x, erp_y)
    face_uv_map = np.stack([erp_x, erp_y], axis=-1)
    
    return face_rgb, face_uv_map
```

#### 3.3.2 Dual Depth Estimation

Run both models and produce two depth representations:

```python
# Step 1: DAP on full equirect → global depth
dap_depth = dap_model.predict(erp_image)  # (H, W) metric meters

# Step 2: Depth Pro on each cubemap face → 6 face depths
face_depths = {}
for face_dir in ['front', 'back', 'left', 'right', 'up', 'down']:
    face_rgb, face_uv = equirect_to_cubemap_face(erp_image, face_dir, fov_deg=100)
    
    # Depth Pro inference (perspective image)
    face_img_pil = Image.fromarray(face_rgb)
    image_tensor, _, f_px = depth_pro.load_rgb(face_img_pil)
    image_tensor = transform(image_tensor)
    prediction = depth_pro_model.infer(image_tensor, f_px=f_px)
    
    face_depths[face_dir] = {
        'depth': prediction['depth'].cpu().numpy(),   # (H_face, W_face) metric meters
        'focal_px': prediction['focallength_px'],
        'uv_map': face_uv
    }
```

#### 3.3.3 Scale-and-Shift Alignment

Depth Pro gives metric depth, but each face has independent scale/shift bias relative to DAP's global prediction. Align using robust least-squares:

```python
def align_face_to_global(face_depth, dap_depth, face_uv_map, 
                          confidence_threshold=0.8):
    """
    Solve: aligned = a * face_depth + b
    Where a, b minimize ||aligned - dap_sampled||^2 over valid pixels.
    
    Uses RANSAC-style robust fitting to handle depth disagreements
    at object boundaries (where DAP is mushy but Depth Pro is sharp).
    """
    # Sample DAP depth at the face's ERP coordinates
    dap_sampled = bilinear_sample(dap_depth, face_uv_map[..., 0], face_uv_map[..., 1])
    
    # Mask out sky/invalid regions
    valid = (face_depth > 0.1) & (dap_sampled > 0.1) & (face_depth < 80) & (dap_sampled < 80)
    
    if valid.sum() < 100:
        return face_depth, 1.0, 0.0  # fallback: no alignment
    
    fd = face_depth[valid].flatten()
    dd = dap_sampled[valid].flatten()
    
    # Robust least-squares (Huber loss or RANSAC)
    # Using median-based estimator for speed:
    ratio = dd / (fd + 1e-8)
    a = np.median(ratio)  # scale
    b = np.median(dd - a * fd)  # shift
    
    aligned = a * face_depth + b
    aligned = np.clip(aligned, 0.01, None)  # no negative depths
    
    return aligned, a, b
```

#### 3.3.4 Reprojection to Equirectangular

After alignment, reproject each face's depth back to the ERP domain using the UV maps, and composite using distance-to-edge weights:

```python
def composite_face_depths(face_results, erp_shape, fov_deg=100):
    """
    Composite 6 aligned face depths back to equirectangular space.
    Uses distance-to-edge weighting for smooth blending in overlap zones.
    """
    H, W = erp_shape
    weight_sum = np.zeros((H, W))
    depth_sum = np.zeros((H, W))
    
    for face_dir, result in face_results.items():
        aligned_depth = result['aligned_depth']
        uv_map = result['uv_map']
        face_size = aligned_depth.shape[0]
        
        # Distance to edge weight (0 at border, 1 at center)
        half = face_size / 2
        # Only the inner 90° core gets full weight; the 5° extension fades to 0
        core_half = face_size * (90 / fov_deg) / 2
        
        u_dist = np.minimum(np.arange(face_size), np.arange(face_size)[::-1])
        v_dist = np.minimum(np.arange(face_size), np.arange(face_size)[::-1])
        uu_dist, vv_dist = np.meshgrid(u_dist, v_dist)
        edge_dist = np.minimum(uu_dist, vv_dist).astype(float)
        
        # Normalize: 0 at face border, 1 at core boundary
        overlap_width = (face_size - face_size * 90 / fov_deg) / 2
        weight = np.clip(edge_dist / max(overlap_width, 1), 0, 1)
        
        # Scatter to ERP
        erp_x = np.round(uv_map[..., 0]).astype(int).clip(0, W-1)
        erp_y = np.round(uv_map[..., 1]).astype(int).clip(0, H-1)
        
        np.add.at(depth_sum, (erp_y, erp_x), aligned_depth * weight)
        np.add.at(weight_sum, (erp_y, erp_x), weight)
    
    # Normalize
    composite = depth_sum / (weight_sum + 1e-8)
    
    # Fill any gaps with DAP depth
    gaps = weight_sum < 0.01
    composite[gaps] = dap_depth[gaps]
    
    return composite
```

#### 3.3.5 Final Fused Depth

The output is a single ERP-space depth map that has:
- DAP's global structure and spherical consistency
- Depth Pro's sharp edges and fine metric detail
- Smooth transitions at face boundaries

```
┌──────────────────────────────────────────────────────────────┐
│                   PHASE 1 PIPELINE                           │
│                                                              │
│  ERP Image ──┬──► DAP ──► Global Depth (low-freq reference)│
│              │                    │                           │
│              ├──► Cubemap Faces (6× 100° FOV)               │
│              │         │                                     │
│              │    Depth Pro (6× inference)                   │
│              │         │                                     │
│              │    Scale/Shift Align ◄── DAP reference        │
│              │         │                                     │
│              │    Feathered Composite                         │
│              │         │                                     │
│              └──► Fused ERP Depth ──► (existing pipeline)   │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 New Module: `spag4d/depth_pro_fusion.py`

```python
class DepthProFusion:
    """
    Cubemap-based Apple Depth Pro integration with DAP global alignment.
    """
    def __init__(self, device='cuda', face_size=1024, fov_deg=100):
        self.device = device
        self.face_size = face_size
        self.fov_deg = fov_deg
        self.depth_pro_model = None
        self.transform = None
    
    def load_model(self):
        """Lazy-load Depth Pro model (~1.5GB VRAM)"""
        import depth_pro
        self.depth_pro_model, self.transform = depth_pro.create_model_and_transforms()
        self.depth_pro_model = self.depth_pro_model.to(self.device).eval()
    
    def fuse(self, erp_image, dap_depth):
        """
        Main entry point. Returns fused ERP depth map.
        
        Args:
            erp_image: (H, W, 3) uint8 equirectangular image
            dap_depth: (H, W) float32 DAP depth in meters
        
        Returns:
            fused_depth: (H, W) float32 fused depth in meters
            confidence: (H, W) float32 confidence map [0, 1]
        """
        if self.depth_pro_model is None:
            self.load_model()
        
        faces = self._project_to_cubemap(erp_image)
        face_depths = self._run_depth_pro(faces)
        aligned_faces = self._align_to_dap(face_depths, dap_depth)
        fused_depth, confidence = self._composite(aligned_faces, dap_depth)
        
        return fused_depth, confidence
    
    def _project_to_cubemap(self, erp_image):
        """Project ERP to 6 extended-FOV cubemap faces."""
        ...
    
    def _run_depth_pro(self, faces):
        """Run Depth Pro inference on each face."""
        ...
    
    def _align_to_dap(self, face_depths, dap_depth):
        """Scale/shift align each face to DAP global depth."""
        ...
    
    def _composite(self, aligned_faces, dap_depth):
        """Feathered composite back to ERP space."""
        ...
```

### 3.5 CLI Integration

```bash
# Enable Depth Pro fusion
python -m spag4d.cli convert panorama.jpg output.ply --depth-pro-fuse

# Control face resolution (higher = more detail, more VRAM)
python -m spag4d.cli convert panorama.jpg output.ply --depth-pro-fuse --face-size 1536

# Combine with existing SHARP
python -m spag4d.cli convert panorama.jpg output.ply --depth-pro-fuse --sharp-refine
```

### 3.6 Icosahedron Option (Alternate Projection)

For completeness, also support icosahedron projection (20 faces) as an alternative to cubemap. The 360MonoDepth project demonstrated that icosahedron tangent faces with 30% padding provide more uniform sphere coverage.

```python
class ProjectionMode(Enum):
    CUBEMAP = "cubemap"       # 6 faces, 100° FOV, fast
    ICOSAHEDRON = "icosa"     # 20 faces, ~70° FOV each, more uniform
    
# CLI:
python -m spag4d.cli convert panorama.jpg output.ply \
    --depth-pro-fuse --projection icosa
```

The icosahedron layout places face centers at the 12 vertices of an icosahedron, producing 20 triangular tangent patches. Each patch is rendered as a square perspective image tangent to the sphere at the face center. With 30% padding, each face covers approximately 70° FOV with substantial overlap.

**Trade-off:** 20 Depth Pro inferences (~6s total at 0.3s each) vs. 6 for cubemap (~1.8s). Use icosahedron when quality matters more than speed.

### 3.7 VRAM Considerations

| Component | VRAM |
|-----------|------|
| DAP model | ~2.5 GB |
| Depth Pro model | ~1.5 GB |
| Face image + depth (1024²) | ~24 MB |
| Total peak (sequential faces) | ~4.5 GB |

Run DAP first, then unload before loading Depth Pro (or keep both if VRAM allows). Sequential face processing means only one face's tensors are in VRAM at a time.

---

## 4. Phase 2: Depth-Aware Gaussian Sizing + Normal-Oriented Covariances

### 4.1 Objective

Replace the uniform isotropic Gaussian parameterization with physically-motivated per-Gaussian sizing and orientation. This is the single biggest visual quality improvement available — it turns blobby point clouds into surface-like renders.

### 4.2 Current Problem

Every Gaussian gets the same `scale_factor` regardless of depth. This means:
- Near-camera objects have splats that are too large → blocky, over-smoothed
- Far-away objects have splats that are too small → holes, transparency
- All splats are spherical → they don't conform to surfaces → visible gaps at grazing angles

### 4.3 Depth-Proportional Gaussian Scale

Each pixel in the equirectangular image subtends a specific solid angle. As depth increases, that solid angle maps to a larger physical area. The Gaussian scale should match this footprint.

```python
def compute_gaussian_scales(depth_map, stride, erp_height, erp_width, 
                             base_scale_factor=1.2):
    """
    Compute per-Gaussian scale proportional to depth and angular pixel footprint.
    
    For an equirectangular image:
      - Horizontal angular resolution: Δφ = 2π / W  (per pixel)
      - Vertical angular resolution:   Δθ = π / H   (per pixel)
      - At latitude θ, the horizontal footprint is: depth * sin(θ) * Δφ * stride
      - The vertical footprint is:                   depth * Δθ * stride
    
    The Gaussian scale should cover the pixel footprint with some overlap.
    """
    H, W = erp_height, erp_width
    dphi = 2 * np.pi / W * stride
    dtheta = np.pi / H * stride
    
    # Build latitude map
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    theta = row_grid / H * np.pi  # 0 at north pole, π at south pole
    
    # Sample depth at stride positions
    sampled_depth = depth_map[::stride, ::stride]
    
    # Horizontal footprint (shrinks near poles due to sin(θ))
    scale_horizontal = sampled_depth * np.sin(theta + 1e-8) * dphi * base_scale_factor
    
    # Vertical footprint (uniform in latitude)
    scale_vertical = sampled_depth * dtheta * base_scale_factor
    
    # Use geometric mean for isotropic approximation,
    # or keep both for anisotropic (Phase 2b)
    scale_isotropic = np.sqrt(scale_horizontal * scale_vertical)
    
    return scale_isotropic, scale_horizontal, scale_vertical
```

### 4.4 Surface Normal Estimation

Compute surface normals from the depth map using finite differences in spherical coordinates. These normals orient the Gaussian covariance to be a flat disc aligned with the surface.

```python
def estimate_normals_from_erp_depth(depth_map, erp_height, erp_width):
    """
    Estimate surface normals from equirectangular depth map.
    
    Uses central finite differences in (θ, φ) space, then converts
    the resulting tangent vectors to Cartesian normals.
    
    Returns:
        normals: (H, W, 3) unit normals in world Cartesian coords
        confidence: (H, W) normal confidence (low at discontinuities)
    """
    H, W = erp_height, erp_width
    
    # Spherical coordinate grids
    theta = np.linspace(0, np.pi, H)     # colatitude
    phi = np.linspace(0, 2*np.pi, W)     # longitude
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert depth to 3D positions
    r = depth_map
    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.cos(theta_grid)
    z = r * np.sin(theta_grid) * np.sin(phi_grid)
    
    # Central differences for tangent vectors
    # dP/dθ (vertical tangent)
    dxdt = np.gradient(x, axis=0)
    dydt = np.gradient(y, axis=0)
    dzdt = np.gradient(z, axis=0)
    tangent_theta = np.stack([dxdt, dydt, dzdt], axis=-1)
    
    # dP/dφ (horizontal tangent) — wraps around
    dxdp = np.gradient(x, axis=1)
    dydp = np.gradient(y, axis=1)
    dzdp = np.gradient(z, axis=1)
    tangent_phi = np.stack([dxdp, dydp, dzdp], axis=-1)
    
    # Normal = cross product of tangent vectors
    normals = np.cross(tangent_theta, tangent_phi)
    normal_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (normal_mag + 1e-8)
    
    # Ensure normals point inward (toward camera at origin)
    positions = np.stack([x, y, z], axis=-1)
    dot = np.sum(normals * positions, axis=-1)
    flip_mask = dot > 0
    normals[flip_mask] *= -1
    
    # Confidence: low where depth gradient is very large (discontinuities)
    depth_grad = np.sqrt(np.gradient(depth_map, axis=0)**2 + 
                         np.gradient(depth_map, axis=1)**2)
    confidence = np.exp(-depth_grad / (np.median(depth_grad) + 1e-8))
    
    return normals, confidence
```

### 4.5 Anisotropic Gaussian Covariance from Normals

Convert the surface normal + scale into a 3D covariance matrix for each Gaussian. The covariance should be a flat ellipsoid aligned with the surface (large in the two tangent directions, thin in the normal direction).

```python
def normal_to_covariance(normal, scale_h, scale_v, thickness_ratio=0.1):
    """
    Build a 3x3 covariance matrix for a surface-aligned Gaussian.
    
    The Gaussian is a flat disc:
      - Two principal axes lie in the tangent plane (scales: scale_h, scale_v)
      - Third axis along the normal (scale: thickness_ratio * min(scale_h, scale_v))
    
    Args:
        normal: (3,) unit normal vector
        scale_h: horizontal scale (meters)
        scale_v: vertical scale (meters)
        thickness_ratio: how thin the disc is (0.1 = 10% of min tangent scale)
    
    Returns:
        covariance: (3, 3) symmetric positive definite matrix
        -OR-
        quaternion: (4,) rotation quaternion + scales: (3,) for 3DGS format
    """
    n = normal / (np.linalg.norm(normal) + 1e-8)
    
    # Build orthonormal frame: find two tangent vectors
    # Pick the axis least aligned with n
    if abs(n[0]) < abs(n[1]) and abs(n[0]) < abs(n[2]):
        up = np.array([1, 0, 0])
    elif abs(n[1]) < abs(n[2]):
        up = np.array([0, 1, 0])
    else:
        up = np.array([0, 0, 1])
    
    t1 = np.cross(n, up)
    t1 = t1 / (np.linalg.norm(t1) + 1e-8)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-8)
    
    # Rotation matrix: columns are the local frame axes
    R = np.stack([t1, t2, n], axis=-1)  # (3, 3)
    
    # Scale along each axis
    s_normal = thickness_ratio * min(scale_h, scale_v)
    S = np.diag([scale_h, scale_v, s_normal])
    
    # Covariance = R @ S² @ R.T
    covariance = R @ (S ** 2) @ R.T
    
    # For 3DGS PLY format, store as quaternion + 3 log-scales
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    # Convert to 3DGS convention [w, x, y, z]
    quat_3dgs = np.array([quat[3], quat[0], quat[1], quat[2]])
    log_scales = np.log(np.array([scale_h, scale_v, s_normal]) + 1e-8)
    
    return quat_3dgs, log_scales
```

### 4.6 Integration into Gaussian Generation

```python
def generate_gaussians(erp_image, depth_map, stride, base_scale_factor=1.2,
                        thickness_ratio=0.1):
    """
    Generate surface-aligned, depth-aware Gaussians.
    
    Replaces the current uniform isotropic generation.
    """
    H, W = depth_map.shape
    
    # Phase 2a: Depth-proportional scales
    scale_iso, scale_h, scale_v = compute_gaussian_scales(
        depth_map, stride, H, W, base_scale_factor
    )
    
    # Phase 2b: Surface normals
    normals, normal_conf = estimate_normals_from_erp_depth(depth_map, H, W)
    normals_sampled = normals[::stride, ::stride]
    normal_conf_sampled = normal_conf[::stride, ::stride]
    scale_h_sampled = scale_h  # already at stride resolution
    scale_v_sampled = scale_v
    
    gaussians = []
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            d = depth_map[row, col]
            if d < depth_min or d > depth_max:
                continue
            
            # 3D position (existing spherical unproject)
            theta = row / H * np.pi
            phi = col / W * 2 * np.pi
            xyz = d * np.array([
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi)
            ])
            
            # RGB
            rgb = erp_image[row, col]
            
            # Oriented covariance
            n = normals_sampled[i, j]
            sh = scale_h_sampled[i, j]
            sv = scale_v_sampled[i, j]
            conf = normal_conf_sampled[i, j]
            
            # Blend between isotropic and anisotropic based on normal confidence
            # Low confidence (depth discontinuities) → more isotropic (safer)
            if conf > 0.3:
                quat, log_scales = normal_to_covariance(n, sh, sv, thickness_ratio)
            else:
                s = scale_iso[i, j]
                quat = np.array([1, 0, 0, 0])  # identity rotation
                log_scales = np.log(np.array([s, s, s * thickness_ratio]) + 1e-8)
            
            opacity = min(conf * 1.2, 1.0)  # modulate by normal confidence
            
            gaussians.append({
                'xyz': xyz,
                'rgb': rgb,
                'quaternion': quat,
                'log_scale': log_scales,
                'opacity': opacity,
            })
    
    return gaussians
```

### 4.7 PLY Format Extension

The current PLY writer needs to include quaternion and anisotropic scale fields. Standard 3DGS PLY format already supports this:

```
property float rot_0    // quaternion w
property float rot_1    // quaternion x
property float rot_2    // quaternion y
property float rot_3    // quaternion z
property float scale_0  // log-scale x
property float scale_1  // log-scale y
property float scale_2  // log-scale z
```

Most viewers (gsplat, SuperSplat, Luma) support these fields natively.

### 4.8 Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Foreground clarity | Blocky, over-smoothed | Crisp, properly scaled |
| Background coverage | Holes, transparent | Filled, large splats |
| Surface rendering | Blobby at grazing angles | Flat, surface-like |
| Gaussian count (same stride) | Same | Same (but better utilized) |

---

## 5. Phase 3: Sky Detection & Pole Thinning

### 5.1 Objective

Remove the two most obvious structural artifacts: garbage geometry in sky regions and artificial density spikes at the poles.

### 5.2 Sky Detection

Sky has no meaningful depth and should not produce Gaussians. Three complementary detection methods:

#### Method A: Depth Threshold (fast, coarse)

```python
def detect_sky_depth(depth_map, depth_max=80.0, threshold_ratio=0.9):
    """
    Pixels at or near depth_max are likely sky.
    """
    sky_mask = depth_map > (depth_max * threshold_ratio)
    return sky_mask
```

#### Method B: Semantic Segmentation (accurate, heavier)

Use a lightweight segmentation model to detect sky pixels. Recommend InternImage-T or SegFormer-B0 for speed.

```python
def detect_sky_semantic(erp_image, model='segformer-b0'):
    """
    Run semantic segmentation to find sky pixels.
    Returns binary mask.
    """
    # Using HuggingFace transformers:
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    # ADE20K sky class = 2
    logits = model(image_tensor).logits
    pred = logits.argmax(dim=1)
    sky_mask = (pred == 2).cpu().numpy()
    return sky_mask
```

#### Method C: Depth Gradient + Color Uniformity (no extra model)

```python
def detect_sky_gradient(depth_map, erp_image, depth_max=80.0):
    """
    Sky regions have:
    - Very high or clamped depth
    - Low depth gradient (flat depth surface)
    - Low color variance
    """
    # High depth
    high_depth = depth_map > depth_max * 0.7
    
    # Low depth gradient
    grad = np.sqrt(np.gradient(depth_map, axis=0)**2 + 
                   np.gradient(depth_map, axis=1)**2)
    low_grad = grad < np.percentile(grad[high_depth] if high_depth.any() else grad, 30)
    
    # Low color variance (in local patches)
    from scipy.ndimage import uniform_filter
    gray = np.mean(erp_image, axis=-1)
    local_mean = uniform_filter(gray, size=32)
    local_sq_mean = uniform_filter(gray**2, size=32)
    local_var = local_sq_mean - local_mean**2
    low_var = local_var < np.percentile(local_var, 20)
    
    sky_mask = high_depth & low_grad & low_var
    return sky_mask
```

#### Sky Handling Options

```python
class SkyMode(Enum):
    SKIP = "skip"               # No Gaussians for sky (transparent void)
    BACKGROUND_SPHERE = "sphere" # Place on a giant sphere at fixed radius
    LOW_OPACITY = "low_opacity"  # Place Gaussians but with very low opacity
```

**Background sphere** mode is recommended for VR/immersive use:

```python
if sky_mode == SkyMode.BACKGROUND_SPHERE:
    sky_radius = depth_max * 2.0  # Fixed large radius
    # Place very large, low-opacity Gaussians on the sphere
    # These form a "skybox" in the splat scene
    for sky_pixel in sky_pixels:
        theta, phi = pixel_to_spherical(sky_pixel)
        xyz = sky_radius * spherical_to_cartesian(theta, phi)
        gaussians.append({
            'xyz': xyz,
            'rgb': erp_image[sky_pixel],
            'scale': sky_radius * 0.05,  # large
            'opacity': 0.3,              # semi-transparent
            'quaternion': [1, 0, 0, 0],  # isotropic
        })
```

### 5.3 Pole Thinning

Equirectangular images oversample the poles. A pixel at the equator covers ~1 steradian, but a pixel at the pole covers ~0 steradians. Without correction, Gaussians pile up at the poles creating a dense blob.

```python
def compute_pole_thinning_mask(H, W, stride, min_density_ratio=0.3):
    """
    Compute a stochastic thinning mask that normalizes Gaussian density
    across the sphere.
    
    At latitude θ, the solid angle per pixel is proportional to sin(θ).
    We thin by randomly dropping pixels with probability 1 - sin(θ).
    
    The min_density_ratio prevents complete elimination at the poles
    (we still want some coverage).
    
    Returns:
        keep_mask: (n_rows, n_cols) boolean mask at stride resolution
    """
    rows = np.arange(0, H, stride)
    theta = rows / H * np.pi  # 0 to π
    
    # sin(θ) is 0 at poles, 1 at equator
    keep_prob = np.sin(theta)
    
    # Floor at min_density_ratio to avoid zero coverage at poles
    keep_prob = np.maximum(keep_prob, min_density_ratio)
    
    # Expand to 2D
    n_cols = len(np.arange(0, W, stride))
    keep_prob_2d = np.tile(keep_prob[:, None], (1, n_cols))
    
    # Stochastic thinning
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    keep_mask = rng.random(keep_prob_2d.shape) < keep_prob_2d
    
    return keep_mask
```

**Alternative: Deterministic latitude-adaptive stride**

Instead of stochastic thinning, increase the effective stride near the poles:

```python
def get_adaptive_stride_per_row(H, base_stride, max_stride_factor=4):
    """
    Returns per-row stride that normalizes Gaussian density.
    At equator: stride = base_stride
    Near poles: stride = base_stride * max_stride_factor
    """
    rows = np.arange(H)
    theta = rows / H * np.pi
    sin_theta = np.sin(theta)
    sin_theta = np.maximum(sin_theta, 1.0 / max_stride_factor)
    
    row_strides = np.round(base_stride / sin_theta).astype(int)
    row_strides = np.clip(row_strides, base_stride, base_stride * max_stride_factor)
    
    return row_strides
```

### 5.4 Combined Filtering Pipeline

```python
def filter_gaussian_candidates(depth_map, erp_image, stride, sky_mode, 
                                 pole_thinning=True):
    """
    Returns a mask of valid Gaussian positions after sky removal
    and pole thinning.
    """
    H, W = depth_map.shape
    
    # Start with all positions valid
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    
    # Sky detection
    sky_mask_erp = detect_sky_gradient(depth_map, erp_image)
    sky_at_stride = sky_mask_erp[::stride, ::stride]
    
    # Depth range filtering
    depth_at_stride = depth_map[::stride, ::stride]
    valid_depth = (depth_at_stride > depth_min) & (depth_at_stride < depth_max)
    
    # Pole thinning
    if pole_thinning:
        pole_mask = compute_pole_thinning_mask(H, W, stride)
    else:
        pole_mask = np.ones_like(sky_at_stride, dtype=bool)
    
    # Combine: valid = not sky AND valid depth AND survived thinning
    if sky_mode == SkyMode.SKIP:
        keep = ~sky_at_stride & valid_depth & pole_mask
    else:
        # Sky pixels handled separately (background sphere or low opacity)
        keep_surface = ~sky_at_stride & valid_depth & pole_mask
        keep_sky = sky_at_stride & pole_mask  # sky still gets pole-thinned
        keep = keep_surface  # sky handled in separate pass
    
    return keep, sky_at_stride
```

### 5.5 Expected Impact

| Issue | Before | After |
|-------|--------|-------|
| Sky regions | Garbage blobby shell at depth_max | Clean void or background sphere |
| North/south poles | Dense splat blob | Uniform density |
| Total Gaussian count | N | ~0.7N (fewer, better-placed) |

---

## 6. Phase 4: Feathered Blending / Laplacian Fusion

### 6.1 Objective

Eliminate visible seams when compositing cubemap face depths, and optimally blend the low-frequency structure of DAP with the high-frequency edges of Depth Pro.

### 6.2 Problem Statement

Phase 1's simple feathered blending handles the cubemap face seams, but the DAP ↔ Depth Pro fusion is still a linear combination. We want: DAP's low frequencies + Depth Pro's high frequencies, without doubling either.

### 6.3 Laplacian Pyramid Depth Fusion

```python
def laplacian_depth_fusion(dap_depth, dp_composite, n_levels=5, 
                            low_freq_cutoff=2):
    """
    Fuse DAP (low-frequency structure) with Depth Pro composite 
    (high-frequency edges) using Laplacian pyramid blending.
    
    Args:
        dap_depth: (H, W) DAP equirectangular depth
        dp_composite: (H, W) Depth Pro composite depth (from Phase 1)
        n_levels: number of pyramid levels
        low_freq_cutoff: levels 0..cutoff use DAP, cutoff+1..n use Depth Pro
    
    Returns:
        fused: (H, W) Laplacian-fused depth
    """
    # Build Gaussian pyramids
    dap_pyramid = build_gaussian_pyramid(dap_depth, n_levels)
    dp_pyramid = build_gaussian_pyramid(dp_composite, n_levels)
    
    # Build Laplacian pyramids
    dap_laplacian = build_laplacian_pyramid(dap_pyramid)
    dp_laplacian = build_laplacian_pyramid(dp_pyramid)
    
    # Fuse: low levels (coarse) from DAP, high levels (fine) from Depth Pro
    fused_laplacian = []
    for level in range(n_levels):
        if level <= low_freq_cutoff:
            # Coarse structure from DAP (equirectangular-aware)
            fused_laplacian.append(dap_laplacian[level])
        else:
            # Fine detail from Depth Pro (sharp boundaries)
            fused_laplacian.append(dp_laplacian[level])
    
    # Reconstruct from fused Laplacian
    fused = reconstruct_from_laplacian(fused_laplacian)
    
    return fused


def build_gaussian_pyramid(image, n_levels):
    """Standard Gaussian pyramid via iterative blur + downsample."""
    pyramid = [image]
    current = image
    for _ in range(n_levels - 1):
        blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    """L[i] = G[i] - upsample(G[i+1])"""
    laplacian = []
    for i in range(len(gaussian_pyramid) - 1):
        upsampled = cv2.resize(gaussian_pyramid[i+1], 
                               (gaussian_pyramid[i].shape[1], 
                                gaussian_pyramid[i].shape[0]))
        laplacian.append(gaussian_pyramid[i] - upsampled)
    # Last level is the residual (lowest frequency)
    laplacian.append(gaussian_pyramid[-1])
    return laplacian


def reconstruct_from_laplacian(laplacian_pyramid):
    """Reconstruct image from Laplacian pyramid."""
    current = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        upsampled = cv2.resize(current, 
                               (laplacian_pyramid[i].shape[1],
                                laplacian_pyramid[i].shape[0]))
        current = upsampled + laplacian_pyramid[i]
    return current
```

### 6.4 Soft Blending with Blend Mask

For more control, use a per-pixel blend mask based on the Depth Pro confidence:

```python
def masked_laplacian_fusion(dap_depth, dp_composite, dp_confidence, 
                             n_levels=5):
    """
    Confidence-weighted Laplacian fusion.
    
    Where Depth Pro is confident (clear surfaces), use its high frequencies.
    Where uncertain (sky, reflections), fall back to DAP entirely.
    """
    # Build pyramids for the blend mask too
    conf_pyramid = build_gaussian_pyramid(dp_confidence, n_levels)
    
    dap_laplacian = build_laplacian_pyramid(build_gaussian_pyramid(dap_depth, n_levels))
    dp_laplacian = build_laplacian_pyramid(build_gaussian_pyramid(dp_composite, n_levels))
    
    fused_laplacian = []
    for level in range(n_levels):
        conf = conf_pyramid[min(level, len(conf_pyramid)-1)]
        if conf.shape != dap_laplacian[level].shape:
            conf = cv2.resize(conf, (dap_laplacian[level].shape[1], 
                                      dap_laplacian[level].shape[0]))
        
        # Weighted blend at each level
        fused_level = conf * dp_laplacian[level] + (1 - conf) * dap_laplacian[level]
        fused_laplacian.append(fused_level)
    
    return reconstruct_from_laplacian(fused_laplacian)
```

### 6.5 Gradient-Domain Blending for Cubemap Seams

As an alternative to the feathered blend in Phase 1, gradient-domain (Poisson) blending produces mathematically optimal seam elimination. This is what 360MonoDepth uses.

```python
def poisson_blend_faces(face_depths, face_uv_maps, target_shape):
    """
    Blend cubemap face depths using Poisson (gradient-domain) blending.
    
    Instead of blending depth values directly, blend depth GRADIENTS,
    then solve for the depth field that best matches all gradients.
    This eliminates seam discontinuities while preserving local detail.
    
    Implementation: Use scipy.sparse.linalg.spsolve with Laplacian system.
    """
    H, W = target_shape
    
    # Compute depth gradients for each face (in ERP space)
    gradient_x = np.zeros((H, W))
    gradient_y = np.zeros((H, W))
    weight_map = np.zeros((H, W))
    
    for face_dir, depth in face_depths.items():
        uv = face_uv_maps[face_dir]
        w = edge_distance_weight(depth.shape[0])  # face edge weights
        
        gx = np.gradient(depth, axis=1)
        gy = np.gradient(depth, axis=0)
        
        # Scatter gradients to ERP
        scatter_add(gradient_x, uv, gx * w)
        scatter_add(gradient_y, uv, gy * w)
        scatter_add(weight_map, uv, w)
    
    gradient_x /= (weight_map + 1e-8)
    gradient_y /= (weight_map + 1e-8)
    
    # Solve Poisson equation: ∇²d = div(gradient)
    divergence = np.gradient(gradient_x, axis=1) + np.gradient(gradient_y, axis=0)
    
    # Sparse linear solve
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import lil_matrix
    
    # ... (standard Poisson solver setup)
    # Boundary condition: use DAP depth as Dirichlet boundary
    
    return solved_depth
```

### 6.6 Recommended Approach

Use **Laplacian pyramid fusion** for the DAP/Depth Pro combination (simpler, fast) and **feathered blending** (Phase 1) for cubemap seams. Reserve Poisson blending as an advanced option for users who need maximum quality.

```python
class BlendMode(Enum):
    FEATHERED = "feathered"      # Simple, fast (Phase 1 default)
    LAPLACIAN = "laplacian"      # Frequency-domain fusion
    POISSON = "poisson"          # Gradient-domain (slowest, best)
```

### 6.7 New Module: `spag4d/depth_blend.py`

```python
class DepthBlender:
    """Handles all depth map fusion and blending operations."""
    
    def fuse_dap_and_depth_pro(self, dap_depth, dp_depth, dp_confidence, 
                                mode='laplacian'):
        """Main fusion entry point."""
        if mode == 'laplacian':
            return laplacian_depth_fusion(dap_depth, dp_depth)
        elif mode == 'masked_laplacian':
            return masked_laplacian_fusion(dap_depth, dp_depth, dp_confidence)
        elif mode == 'linear':
            # Simple fallback
            alpha = 0.5
            return alpha * dp_depth + (1 - alpha) * dap_depth
    
    def blend_cubemap_faces(self, faces, uv_maps, target_shape, 
                             mode='feathered'):
        """Cubemap face compositing."""
        if mode == 'feathered':
            return composite_face_depths(faces, target_shape)
        elif mode == 'poisson':
            return poisson_blend_faces(faces, uv_maps, target_shape)
```

---

## 7. Phase 5: Adaptive Stride

### 7.1 Objective

Allocate the Gaussian budget more efficiently by using smaller stride (more Gaussians) in nearby/detailed regions and larger stride (fewer Gaussians) in distant/uniform regions.

### 7.2 Depth-Based Adaptive Stride

The core idea: closer objects need more Gaussians because they subtend a larger visual angle and show more detail. Far objects can be represented with fewer, larger splats.

```python
def compute_adaptive_stride_map(depth_map, base_stride=2, 
                                  min_stride=1, max_stride=8,
                                  depth_reference=5.0):
    """
    Compute per-pixel stride based on depth.
    
    Stride is proportional to depth: close objects get stride=min_stride,
    far objects get stride=max_stride.
    
    Args:
        depth_map: (H, W) depth in meters
        base_stride: stride at depth_reference distance
        min_stride: minimum stride (most dense)
        max_stride: maximum stride (least dense)
        depth_reference: depth at which stride = base_stride
    
    Returns:
        stride_map: (H, W) integer stride values
    """
    # Linear mapping: stride = base_stride * (depth / depth_reference)
    stride_float = base_stride * (depth_map / depth_reference)
    stride_int = np.round(stride_float).astype(int)
    stride_int = np.clip(stride_int, min_stride, max_stride)
    
    return stride_int
```

### 7.3 Implementation: Variable-Stride Sampling

Variable stride can't use simple `[::stride, ::stride]` indexing. Instead, use a greedy row-by-row sampler:

```python
def sample_with_adaptive_stride(depth_map, erp_image, normals, 
                                  stride_map, sky_mask, pole_mask):
    """
    Sample Gaussian positions using per-pixel adaptive stride.
    
    Uses a greedy approach: walk each row, skipping by the local stride.
    
    Returns:
        positions: list of (row, col) tuples
    """
    H, W = depth_map.shape
    positions = []
    
    # Row stride: use median stride in each row (for pole-aware spacing)
    row = 0
    while row < H:
        # Determine row-level stride from median of this row's stride map
        row_stride = int(np.median(stride_map[row, :]))
        row_stride = max(row_stride, 1)
        
        col = 0
        while col < W:
            if not sky_mask[row, col] and pole_mask[row // stride_map[row, col], 
                                                      col // stride_map[row, col]]:
                local_stride = stride_map[row, col]
                positions.append((row, col))
                col += local_stride
            else:
                col += stride_map[row, col]
        
        row += row_stride
    
    return positions
```

### 7.4 Edge-Aware Stride Refinement

Additionally, reduce stride near depth discontinuities (object edges) to capture boundary detail:

```python
def refine_stride_at_edges(stride_map, depth_map, edge_reduction_factor=0.5):
    """
    Reduce stride near depth edges to capture boundary detail.
    
    Detects edges via Canny on the depth map, then dilates the edge mask
    and reduces stride in those regions.
    """
    # Normalize depth to 0-255 for Canny
    depth_norm = ((depth_map - depth_map.min()) / 
                  (depth_map.max() - depth_map.min() + 1e-8) * 255).astype(np.uint8)
    
    edges = cv2.Canny(depth_norm, 30, 100)
    
    # Dilate edges to create a border zone
    kernel = np.ones((7, 7), dtype=np.uint8)
    edge_zone = cv2.dilate(edges, kernel, iterations=2) > 0
    
    # Reduce stride in edge zones
    refined = stride_map.copy()
    refined[edge_zone] = np.maximum(
        (refined[edge_zone] * edge_reduction_factor).astype(int), 
        1
    )
    
    return refined
```

### 7.5 Gaussian Budget Control

Users should be able to specify a target Gaussian count rather than a stride:

```python
def compute_stride_for_budget(depth_map, target_count, sky_mask,
                               min_stride=1, max_stride=8):
    """
    Binary search for the base_stride that produces approximately
    target_count Gaussians after sky removal and pole thinning.
    """
    H, W = depth_map.shape
    valid_pixels = (~sky_mask).sum()
    
    # Approximate: count ≈ valid_pixels / stride²
    # (Pole thinning reduces by ~30%, so adjust)
    approx_stride = np.sqrt(valid_pixels * 0.7 / target_count)
    approx_stride = int(np.clip(approx_stride, min_stride, max_stride))
    
    # Binary search refinement
    # ...
    
    return approx_stride
```

### 7.6 CLI Integration

```bash
# Adaptive stride (automatic)
python -m spag4d.cli convert panorama.jpg output.ply --adaptive-stride

# Target Gaussian count
python -m spag4d.cli convert panorama.jpg output.ply --target-gaussians 500000

# Fixed stride with edge refinement
python -m spag4d.cli convert panorama.jpg output.ply --stride 2 --edge-refine
```

### 7.7 Expected Impact

| Metric | Uniform stride=2 | Adaptive stride |
|--------|-------------------|-----------------|
| Foreground density | Under-sampled | Dense where needed |
| Background density | Over-sampled | Sparse, efficient |
| Edge coverage | Random | Dense at boundaries |
| Total count | N | ~N (redistributed) |

---

## 8. Phase 6: Tangent Patch Projection (Upgrade Path)

### 8.1 Objective

Replace cubemap projection with overlapping tangent patches (à la 360MonoDepth) for maximum depth quality. This is the "no compromises" path when cubemap seams remain problematic.

### 8.2 Why Tangent Patches?

Cubemap has three problems that tangent patches solve:
1. **Top/bottom face ambiguity:** The up/down cubemap faces have 4-fold rotational symmetry, confusing perspective depth models.
2. **Only 6 faces:** Limited overlap zones (only at edges). If one face's depth is wrong, there's no redundancy.
3. **Non-uniform coverage:** Face centers get better coverage than corners.

Tangent patches (icosahedron or custom sampling) provide uniform coverage, dense overlap for alignment, and avoid the pole-face problem.

### 8.3 Icosahedron Layout

Following 360MonoDepth, use 20 tangent images centered at icosahedron face centers:

```python
# Icosahedron face centers (unit sphere coordinates)
ICOSA_VERTICES = [
    # ... 12 vertices of regular icosahedron
]
ICOSA_FACE_CENTERS = [
    # ... 20 face centers (centroid of each triangular face)
    # Pre-computed and stored as (θ, φ) pairs
]

class TangentPatchProjector:
    """
    Projects ERP image to/from icosahedron tangent patches.
    Based on 360MonoDepth with modifications for depth fusion.
    """
    def __init__(self, n_patches=20, face_size=512, padding=0.3):
        """
        Args:
            n_patches: 20 for icosahedron, 6 for cubemap
            face_size: output resolution per patch
            padding: extend each patch by this fraction (0.3 = 30%)
        """
        self.n_patches = n_patches
        self.face_size = face_size
        self.padding = padding
        
        if n_patches == 20:
            self.centers = ICOSA_FACE_CENTERS
            self.base_fov = 70  # degrees (before padding)
        elif n_patches == 6:
            self.centers = CUBEMAP_FACE_CENTERS
            self.base_fov = 90
        
        self.effective_fov = self.base_fov * (1 + padding)
    
    def project_forward(self, erp_image):
        """
        Project ERP image to all tangent patches.
        
        Returns:
            patches: list of (face_size, face_size, 3) images
            uv_maps: list of (face_size, face_size, 2) ERP coordinate maps
        """
        patches = []
        uv_maps = []
        
        for center_theta, center_phi in self.centers:
            patch, uv = self._project_single_patch(
                erp_image, center_theta, center_phi
            )
            patches.append(patch)
            uv_maps.append(uv)
        
        return patches, uv_maps
    
    def _project_single_patch(self, erp_image, center_theta, center_phi):
        """
        Gnomonic projection of ERP onto a tangent plane at (θ, φ).
        
        The tangent plane touches the unit sphere at the center point.
        Pixels are sampled from the ERP image using the inverse mapping.
        """
        H, W = erp_image.shape[:2]
        fov_rad = np.radians(self.effective_fov)
        
        # Build pixel grid on tangent plane
        half_size = np.tan(fov_rad / 2)
        u = np.linspace(-half_size, half_size, self.face_size)
        v = np.linspace(-half_size, half_size, self.face_size)
        uu, vv = np.meshgrid(u, v)
        
        # 3D points on tangent plane (in local frame)
        points_local = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        
        # Rotate to world frame (align local z-axis with center direction)
        R = self._rotation_to_center(center_theta, center_phi)
        points_world = points_local @ R.T
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points_world, axis=-1, keepdims=True)
        points_sphere = points_world / norms
        
        # Convert to ERP coordinates
        theta = np.arccos(np.clip(points_sphere[..., 1], -1, 1))
        phi = np.arctan2(points_sphere[..., 0], points_sphere[..., 2])
        
        erp_x = ((phi + np.pi) / (2 * np.pi)) * W
        erp_y = (theta / np.pi) * H
        
        # Bilinear sample
        patch = bilinear_sample(erp_image, erp_x, erp_y)
        uv_map = np.stack([erp_x, erp_y], axis=-1)
        
        return patch, uv_map
    
    def _rotation_to_center(self, theta, phi):
        """Rotation matrix that aligns z-axis with the center direction."""
        # Center direction in Cartesian
        cx = np.sin(theta) * np.cos(phi)
        cy = np.cos(theta)
        cz = np.sin(theta) * np.sin(phi)
        center = np.array([cx, cy, cz])
        
        # Build orthonormal frame
        up = np.array([0, 1, 0])
        if abs(np.dot(center, up)) > 0.99:
            up = np.array([1, 0, 0])
        
        right = np.cross(up, center)
        right /= np.linalg.norm(right)
        true_up = np.cross(center, right)
        
        R = np.stack([right, true_up, center], axis=-1)  # columns
        return R
```

### 8.4 Multi-Patch Depth Fusion

With 20 overlapping patches, the fusion strategy is:

1. **Run Depth Pro on each patch** (20 inferences)
2. **Pairwise scale alignment** in overlap zones
3. **Global optimization** to find per-patch scale/shift that minimizes discrepancy
4. **Gradient-domain blending** to composite back to ERP

```python
class TangentDepthFusion:
    """
    Fuse depth predictions from overlapping tangent patches.
    Follows 360MonoDepth: deformable alignment + gradient-domain blending.
    """
    
    def fuse_patches(self, patch_depths, uv_maps, dap_depth, erp_shape):
        """
        Main fusion pipeline.
        
        Args:
            patch_depths: list of (face_size, face_size) depth maps
            uv_maps: list of ERP coordinate maps
            dap_depth: (H, W) DAP global depth for scale reference
            erp_shape: (H, W) output shape
        """
        # Step 1: Align each patch to DAP
        aligned = []
        for i, (pd, uv) in enumerate(zip(patch_depths, uv_maps)):
            a_pd, scale, shift = align_face_to_global(pd, dap_depth, uv)
            aligned.append(a_pd)
        
        # Step 2: Pairwise refinement in overlap zones
        aligned = self._pairwise_refine(aligned, uv_maps)
        
        # Step 3: Global least-squares scale optimization
        aligned = self._global_optimize(aligned, uv_maps, dap_depth)
        
        # Step 4: Gradient-domain blending
        fused = self._gradient_blend(aligned, uv_maps, erp_shape)
        
        return fused
    
    def _pairwise_refine(self, patches, uv_maps):
        """
        For each pair of overlapping patches, compute relative 
        scale/shift and adjust to minimize discrepancy.
        """
        n = len(patches)
        # Build overlap graph
        for i in range(n):
            for j in range(i+1, n):
                overlap = compute_overlap(uv_maps[i], uv_maps[j])
                if overlap.sum() > 50:  # sufficient overlap
                    # Align j to i in overlap zone
                    ratio = np.median(
                        patches[i][overlap] / (patches[j][overlap] + 1e-8)
                    )
                    # Store edge weight for global optimization
                    ...
        return patches
    
    def _global_optimize(self, patches, uv_maps, dap_depth):
        """
        Solve for per-patch (scale, shift) that minimizes:
          Σ_edges ||a_i * d_i + b_i - a_j * d_j - b_j||² in overlap
          + λ * Σ_patches ||a_i * d_i + b_i - dap_sampled||²
        """
        # Linear system: 2N unknowns (a_i, b_i for each patch)
        # Sparse matrix construction + solve
        ...
        return optimized_patches
    
    def _gradient_blend(self, patches, uv_maps, erp_shape):
        """
        Poisson blending: composite depth gradients, then solve for 
        the depth field.
        """
        # Same as Phase 4 Poisson blend but with 20 patches
        ...
        return fused_depth
```

### 8.5 Non-Uniform Patch Sampling (Advanced)

Instead of fixed icosahedron centers, adaptively place more patches in complex regions:

```python
def adaptive_patch_placement(erp_image, base_n=20, max_n=40):
    """
    Place more patches in high-complexity regions (detected by 
    edge density in the input image).
    
    1. Start with icosahedron 20 patches
    2. Compute "complexity" per patch (edge density, texture variance)
    3. Subdivide high-complexity patches (add extra centers)
    """
    # Start with standard layout
    centers = ICOSA_FACE_CENTERS.copy()
    
    # Compute complexity map on ERP
    gray = cv2.cvtColor(erp_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # For each existing center, measure edge density in its FOV
    complexities = []
    for theta, phi in centers:
        mask = get_patch_footprint(theta, phi, erp_image.shape[:2])
        complexity = edges[mask].mean()
        complexities.append(complexity)
    
    # Add extra patches for high-complexity regions
    threshold = np.percentile(complexities, 75)
    for i, (theta, phi) in enumerate(centers[:]):
        if complexities[i] > threshold and len(centers) < max_n:
            # Add 3 sub-patches around this center
            for dtheta, dphi in [(0.2, 0), (-0.2, 0), (0, 0.2)]:
                centers.append((theta + dtheta, phi + dphi))
    
    return centers[:max_n]
```

### 8.6 Performance Comparison

| Method | Patches | Inference Time | Seam Quality | Pole Handling |
|--------|---------|----------------|--------------|---------------|
| Cubemap | 6 | ~1.8s | Good (overlap blend) | Poor (top/bottom face) |
| Icosahedron | 20 | ~6s | Excellent (dense overlap) | Good (uniform) |
| Adaptive | 20-40 | ~6-12s | Excellent | Good |

---

## 9. New Module Map

```
spag4d/
├── core.py                     # Main orchestrator (updated)
├── cli.py                      # CLI interface (updated)
├── dap_arch/                   # DAP model wrapper (unchanged)
│   └── DAP/                    # DAP repository
├── sharp_refiner.py            # SHARP integration (unchanged)
│
├── depth_pro_fusion.py         # [NEW] Phase 1: Cubemap Depth Pro
│   ├── DepthProFusion          #   Main fusion class
│   ├── equirect_to_cubemap_face()
│   ├── align_face_to_global()
│   └── composite_face_depths()
│
├── depth_blend.py              # [NEW] Phase 4: Laplacian/Poisson blending
│   ├── DepthBlender
│   ├── laplacian_depth_fusion()
│   ├── masked_laplacian_fusion()
│   └── poisson_blend_faces()
│
├── tangent_projector.py        # [NEW] Phase 6: Tangent patch projection
│   ├── TangentPatchProjector
│   ├── TangentDepthFusion
│   └── adaptive_patch_placement()
│
├── gaussian_params.py          # [NEW] Phase 2: Gaussian parameterization
│   ├── compute_gaussian_scales()
│   ├── estimate_normals_from_erp_depth()
│   ├── normal_to_covariance()
│   └── generate_gaussians()
│
├── scene_filter.py             # [NEW] Phase 3: Sky/pole filtering
│   ├── detect_sky_gradient()
│   ├── detect_sky_semantic()
│   ├── compute_pole_thinning_mask()
│   └── filter_gaussian_candidates()
│
├── adaptive_stride.py          # [NEW] Phase 5: Adaptive stride
│   ├── compute_adaptive_stride_map()
│   ├── refine_stride_at_edges()
│   ├── sample_with_adaptive_stride()
│   └── compute_stride_for_budget()
│
├── ply_writer.py               # [UPDATED] Support quaternion + aniso scale
└── splat_writer.py             # [UPDATED] Support quaternion + aniso scale
```

### Updated `core.py` Pipeline

```python
class SPAG4D:
    def convert(self, input_path, output_path, **kwargs):
        # 1. Load equirectangular image
        erp_image = load_image(input_path)
        
        # 2. DAP depth estimation (always)
        dap_depth = self.dap_model.predict(erp_image)
        
        # 3. Optional: Depth Pro fusion (Phase 1)
        if kwargs.get('depth_pro_fuse'):
            projection = kwargs.get('projection', 'cubemap')
            if projection == 'cubemap':
                fused_depth, confidence = self.depth_pro_fusion.fuse(
                    erp_image, dap_depth
                )
            elif projection == 'icosa':
                fused_depth, confidence = self.tangent_fusion.fuse(
                    erp_image, dap_depth
                )
            
            # Phase 4: Laplacian blend DAP + fused
            blend_mode = kwargs.get('blend_mode', 'laplacian')
            depth = self.blender.fuse_dap_and_depth_pro(
                dap_depth, fused_depth, confidence, mode=blend_mode
            )
        else:
            depth = dap_depth
        
        # 4. Optional: SHARP refinement (existing)
        if kwargs.get('sharp_refine'):
            depth = self.sharp_refiner.refine(depth, erp_image)
        
        # 5. Phase 3: Sky detection + pole thinning
        keep_mask, sky_mask = filter_gaussian_candidates(
            depth, erp_image, 
            stride=kwargs.get('stride', 2),
            sky_mode=kwargs.get('sky_mode', SkyMode.SKIP),
            pole_thinning=kwargs.get('pole_thinning', True)
        )
        
        # 6. Phase 5: Adaptive stride (optional)
        if kwargs.get('adaptive_stride'):
            stride_map = compute_adaptive_stride_map(depth)
            stride_map = refine_stride_at_edges(stride_map, depth)
            positions = sample_with_adaptive_stride(
                depth, erp_image, normals, stride_map, sky_mask, keep_mask
            )
        else:
            positions = uniform_positions(depth.shape, kwargs['stride'], keep_mask)
        
        # 7. Phase 2: Depth-aware Gaussian generation
        normals, normal_conf = estimate_normals_from_erp_depth(depth, *depth.shape)
        gaussians = generate_gaussians(
            erp_image, depth, positions, normals, normal_conf,
            base_scale_factor=kwargs.get('scale_factor', 1.2),
            thickness_ratio=kwargs.get('thickness', 0.1)
        )
        
        # 8. Write output
        write_ply(output_path, gaussians)  # with quaternion + aniso scale
```

---

## 10. CLI / API Surface Changes

### New CLI Arguments

```
Depth Pro Fusion (Phase 1):
  --depth-pro-fuse        Enable Depth Pro cubemap fusion
  --projection {cubemap,icosa}  Projection method (default: cubemap)
  --face-size SIZE        Face resolution in pixels (default: 1024)

Gaussian Quality (Phase 2):
  --oriented-gaussians    Enable normal-oriented covariance (default: on)
  --thickness RATIO       Gaussian thickness ratio (default: 0.1)

Scene Filtering (Phase 3):
  --sky-mode {skip,sphere,low_opacity}  Sky handling (default: skip)
  --sky-radius METERS     Background sphere radius (default: 200)
  --no-pole-thinning      Disable pole density normalization

Blending (Phase 4):
  --blend-mode {feathered,laplacian,poisson}  Depth fusion method
  --blend-levels N        Laplacian pyramid levels (default: 5)

Adaptive Stride (Phase 5):
  --adaptive-stride       Enable depth-proportional stride
  --target-gaussians N    Target Gaussian count (auto-computes stride)
  --edge-refine           Dense sampling at depth edges
```

### Python API

```python
converter = SPAG4D(
    device='cuda',
    use_depth_pro=True,       # loads Depth Pro model
    use_sharp_refinement=True  # existing
)

result = converter.convert(
    input_path='panorama.jpg',
    output_path='output.ply',
    
    # Phase 1
    depth_pro_fuse=True,
    projection='cubemap',
    
    # Phase 2
    oriented_gaussians=True,
    thickness=0.1,
    
    # Phase 3
    sky_mode='sphere',
    pole_thinning=True,
    
    # Phase 4
    blend_mode='laplacian',
    
    # Phase 5
    adaptive_stride=True,
    target_gaussians=500_000,
)
```

---

## 11. VRAM Budget & Performance

### Per-Phase VRAM Requirements

| Phase | New VRAM | Cumulative Peak | Notes |
|-------|----------|-----------------|-------|
| Baseline (DAP) | ~2.5 GB | 2.5 GB | Current |
| Phase 1 (Depth Pro) | ~1.5 GB | ~4.0 GB | Sequential: unload DAP, load DP |
| Phase 2 (Normals) | ~0.2 GB | ~4.2 GB | NumPy on CPU, minimal GPU |
| Phase 3 (Sky/Poles) | ~0.1 GB | ~4.3 GB | CPU-only filtering |
| Phase 4 (Blending) | ~0.3 GB | ~4.6 GB | Pyramid in RAM |
| Phase 5 (Adaptive) | ~0.1 GB | ~4.7 GB | CPU stride computation |
| Phase 6 (Tangent) | ~1.5 GB | ~4.5 GB | Replaces Phase 1 cubemap |

**Recommendation:** 8GB VRAM is sufficient if models are loaded sequentially. 12GB+ allows concurrent DAP + Depth Pro for faster processing.

### Processing Time Estimates (4K ERP input, RTX 3080)

| Phase | Added Time | Total Pipeline |
|-------|-----------|----------------|
| Baseline | ~5s | 5s |
| Phase 1 (cubemap) | +3s | 8s |
| Phase 1 (icosa) | +8s | 13s |
| Phase 2 | +0.5s | 8.5s |
| Phase 3 | +0.3s | 8.8s |
| Phase 4 | +1s | 9.8s |
| Phase 5 | +0.2s | 10s |
| Phase 6 (replaces 1) | +8s | 15s |

---

## 12. Testing Strategy

### Visual Quality Metrics

Since SPAG-4D operates from single panoramas (no ground truth 3D), testing must focus on perceptual quality and consistency:

1. **Seam visibility test:** Render the splat scene from 6 views that look directly at cubemap/patch boundaries. Score visibility of depth discontinuities.

2. **Pole density test:** Render top-down and bottom-up views. The Gaussian density should be roughly uniform, not a bright blob.

3. **Sky cleanliness test:** Render with a contrasting background. No stray Gaussians should float in sky regions.

4. **Edge sharpness test:** Render close-up views of object boundaries. Compare DAP-only vs. Depth Pro fusion.

5. **Scale consistency test:** Render at varying distances. Nearby objects should be detailed, far objects should be coherent (no holes, no bloat).

### Automated Regression Tests

```python
# tests/test_quality.py

def test_gaussian_count_within_budget():
    """Target gaussian count matches within 10%."""
    result = converter.convert(..., target_gaussians=500_000)
    assert 450_000 < result.splat_count < 550_000

def test_no_sky_gaussians():
    """No gaussians placed at depth_max in sky regions."""
    result = converter.convert(..., sky_mode='skip')
    # Check no xyz positions are at exactly depth_max radius
    radii = np.linalg.norm(result.positions, axis=-1)
    assert (radii > depth_max * 0.95).sum() == 0

def test_pole_density_uniform():
    """Gaussian density per steradian is roughly uniform."""
    theta = np.arccos(result.positions[:, 1] / radii)
    # Bin by latitude
    bins = np.linspace(0, np.pi, 18)  # 10° bins
    counts, _ = np.histogram(theta, bins)
    # Weight by solid angle
    solid_angles = np.diff(-np.cos(bins))
    density = counts / solid_angles
    # Coefficient of variation < 0.5
    assert density.std() / density.mean() < 0.5

def test_depth_pro_alignment():
    """Depth Pro face depths align with DAP within 10% median error."""
    # Run alignment, check residuals
    ...

def test_normal_orientation():
    """Gaussian normals point inward (toward origin)."""
    dots = np.sum(result.normals * result.positions, axis=-1)
    assert (dots < 0).mean() > 0.95  # >95% point inward

def test_ply_format_valid():
    """Output PLY has quaternion + anisotropic scale fields."""
    import plyfile
    ply = plyfile.PlyData.read('output.ply')
    vertex = ply['vertex']
    assert 'rot_0' in vertex.data.dtype.names
    assert 'scale_0' in vertex.data.dtype.names
```

### Test Panoramas

Include a curated set of test images covering:
- Indoor scene (room with furniture)
- Outdoor landscape (mountains + sky)
- Urban street (buildings + ground)
- Night scene (challenging depth)
- High-dynamic-range scene (bright sky + dark interior)

These should be added to `TestImage/` with expected baseline metrics.

---

## 13. References

1. **DAP - Depth Any Panoramas** — Insta360 Research. Equirectangular-aware depth estimation. https://github.com/Insta360-Research-Team/DAP

2. **Apple Depth Pro** — Bochkovskii et al. "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second." Zero-shot metric depth with sharp boundaries. https://github.com/apple/ml-depth-pro

3. **360MonoDepth** — Rey-Area et al. (CVPR 2022) "360MonoDepth: High-Resolution 360° Monocular Depth Estimation." Tangent patch projection with deformable alignment and gradient-domain blending. https://github.com/manurare/360monodepth

4. **OmniFusion** — Li et al. (2022) "OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion." Geometry-aware feature fusion for tangent patches.

5. **BiFuse / UniFuse** — Wang et al. (CVPR 2020) Bi-projection fusion of equirectangular and cubemap for 360° depth.

6. **SphereFusion** — Equirectangular + spherical mesh dual-encoder with gated fusion.

7. **3D Gaussian Splatting** — Kerbl et al. (SIGGRAPH 2023) "3D Gaussian Splatting for Real-Time Radiance Field Rendering." https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

8. **SuGaR** — Guédon & Lepetit (2023) "SuGaR: Surface-Aligned Gaussian Splatting." Surface regularization for Gaussians.

9. **Cross360** — (2025) "Cross360: 360° Monocular Depth Estimation via Cross Projections Across Scales." Cross-projection feature alignment between ERP and tangent patches.

10. **CAPDepth** — (2025) Content-aware projection with Pannini optimization for reduced tangent distortion.

---

## Appendix A: Phase Implementation Order & Dependencies

```
Phase 1 ─────► Phase 4 (blending improves Phase 1 output)
   │
   └──────────► Phase 6 (tangent patches replace cubemap)

Phase 2 ─────► standalone (no dependencies)

Phase 3 ─────► standalone (no dependencies)

Phase 5 ─────► depends on Phase 3 (sky mask needed for budget calc)
```

**Recommended sprint plan:**
- Sprint 1: Phase 2 + Phase 3 (biggest quality wins, no new models)
- Sprint 2: Phase 1 (Depth Pro integration)
- Sprint 3: Phase 4 + Phase 5 (refinements)
- Sprint 4: Phase 6 (upgrade path, optional)

---

*End of document.*
