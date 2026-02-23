# SPAG-4D: Gaussian Splat Output Quality Improvements

## Codebase Architecture Summary

The pipeline flows: **ERP Image → Depth Estimation (PanDA/DAP/DA3) → Guided Depth Refinement → SHARP Attribute Refinement → Spherical Grid → Gaussian Conversion → PLY/SPLAT Export**.

After reviewing every module, here are the quality bottlenecks and improvement opportunities, ranked roughly by impact.

---

## 1. Depth Estimation — The Biggest Single Lever

### Problem: Linear Relative-to-Metric Scaling (panda_model.py:270-281)

PanDA outputs *relative* depth (0–1) and the code maps it with a **linear** scale:

```python
depth = self.depth_min + depth_normalized * (self.depth_max - self.depth_min)
```

Real-world depth distributions are log-normal, not linear. A linear mapping compresses all foreground detail into a tiny range while the 0.1–100m span is dominated by background. This makes nearby objects (0.1–5m) nearly indistinguishable from each other while wasting dynamic range on far-field values.

### Fix: Log-space or Inverse-Depth Mapping

```python
# Log-space mapping preserves foreground detail
log_min = math.log(self.depth_min)
log_max = math.log(self.depth_max)
depth = torch.exp(log_min + depth_normalized * (log_max - log_min))
```

Or even better, use **inverse depth** (disparity) mapping, which is what stereo and monocular depth models are actually trained to estimate:

```python
# Inverse-depth mapping (disparity-linear)
inv_max = 1.0 / self.depth_min   # close = large disparity
inv_min = 1.0 / self.depth_max   # far = small disparity
inv_depth = inv_max - depth_normalized * (inv_max - inv_min)
depth = 1.0 / inv_depth.clamp(min=1e-6)
```

**Impact**: High. This single change would dramatically improve the depth separation of foreground objects and reduce the "flattened cardboard" look that monocular-to-splat pipelines often exhibit.

### Problem: Resolution Capping (panda_model.py:217-223)

```python
MAX_INPUT_WIDTH = 1022
```

PanDA is forced to process at ≤1022px width regardless of input resolution. A 4K or 8K panorama gets downsampled before depth estimation, losing fine-grained depth edges that no amount of post-filtering can recover.

### Fix: Tiled/Overlapping Inference

Process the panorama in overlapping vertical strips (since ERP is horizontally wrappable) and stitch the depth. Alternatively, run PanDA at multiple scales and fuse:

```python
# Multi-scale depth fusion
scales = [1.0, 0.5] if W > 2048 else [1.0]
depths = []
for s in scales:
    resized = F.interpolate(x, scale_factor=s, mode='bilinear')
    d = self.model(resized)
    d_upsampled = F.interpolate(d, size=(H, W), mode='bilinear')
    depths.append(d_upsampled)
depth = torch.stack(depths).mean(dim=0)  # or weighted by confidence
```

**Impact**: High for high-res inputs. Recovers fine geometry that stride-2 downsampling in `gaussian_converter.py` currently obliterates.

---

## 2. Gaussian Scale and Opacity Model — Uniformity Problem

### Problem: Constant Default Opacity (gaussian_converter.py:26, 150)

Every Gaussian gets `default_opacity = 0.95`. This creates a uniformly opaque shell with no transparency variation. Real scenes have semi-transparent elements (foliage, glass, thin structures) that should have lower opacity, and dense surfaces that should be fully opaque.

### Fix: Depth-Gradient-Adaptive Opacity

Compute opacity based on local depth consistency — areas with smooth depth are likely surfaces (high opacity), areas with high depth variance are likely edges or thin structures (lower opacity):

```python
# Compute local depth gradient magnitude
dx = depth[:, 1:] - depth[:, :-1]
dy = depth[1:, :] - depth[:-1, :]
# Pad to match dimensions
grad_mag = torch.sqrt(F.pad(dx, (0,1))**2 + F.pad(dy, (1,0))**2)
grad_mag = grad_mag / (depth.clamp(min=0.1))  # Normalize by depth

# Smooth surfaces → high opacity, edges → lower opacity
opacity = 0.99 - 0.5 * torch.sigmoid(grad_mag * 10 - 3)
```

**Impact**: Medium-high. Reduces the "paper cutout" look at object edges and depth discontinuities.

### Problem: Uniform Scale Factor (gaussian_converter.py:113-116)

The Gaussian scale is purely geometric (`scale_factor * depth * angular_extent * stride`). There's no adaptation to local image content — a textured brick wall gets the same scale as a smooth sky region.

### Fix: Content-Adaptive Scale

Use local image gradient to modulate scale — high-frequency regions (texture, edges) get smaller Gaussians for detail, smooth regions get larger ones for efficiency:

```python
# Image gradient magnitude (per-channel, take max)
img_gray = colors.mean(dim=-1)
gx = (img_gray[:, 1:] - img_gray[:, :-1]).abs()
gy = (img_gray[1:, :] - img_gray[:-1, :]).abs()
gradient = torch.sqrt(F.pad(gx, (0,1))**2 + F.pad(gy, (1,0))**2)

# Scale multiplier: high gradient → smaller Gaussians (more detail)
detail_factor = 1.0 / (1.0 + gradient * 5.0)  # range ~[0.17, 1.0]
detail_factor = detail_factor.clamp(0.3, 1.0)

s_azimuth = scale_factor * depth * sin_phi * delta_theta * stride * detail_factor
s_elevation = scale_factor * depth * delta_phi * stride * detail_factor
```

**Impact**: Medium. Lets the splat viewer resolve fine details in textured areas without over-densifying smooth regions.

---

## 3. Depth Edge Quality — Guided Filter Limitations

### Problem: Global Min/Max Rescaling (depth_refiner.py:88-93)

After guided filtering, the depth range is globally rescaled:

```python
result = (result - result_min) / (result_max - result_min)
result = result * (orig_max - orig_min) + orig_min
```

This destroys the local edge refinement that the guided filter was supposed to provide. If the filter sharpens one edge, the global rescaling shifts *all* other depths to compensate.

### Fix: Preserve Absolute Values, Only Blend

```python
# Don't rescale globally — the guided filter already preserves structure
# Just blend between original and filtered
if strength < 1.0:
    result = result * strength + depth * (1.0 - strength)
```

Or better yet, operate in log-depth space where edges are more perceptually meaningful:

```python
depth_log = torch.log1p(depth)
guide_np = rgb_guide.cpu().numpy()
depth_log_np = depth_log.cpu().numpy()
result_log = guided_filter(depth_log_np, guide_np)
result = torch.expm1(torch.from_numpy(result_log).to(device))
```

**Impact**: Medium. Fixes a subtle but real issue where guided filtering currently has less effect than intended.

### Problem: Grayscale-Only Guide (depth_refiner.py:152-153)

The Python fallback converts the RGB guide to grayscale before filtering:

```python
guide_gray = np.mean(guide, axis=2).astype(np.float32)
```

This loses chrominance edges — a red object on a green background at the same luminance won't get depth edge sharpening.

### Fix: Multi-Channel Guided Filter

Use the full color guided filter (He et al. 2013 "Fast Guided Filter"). The OpenCV path already handles this, but the Python fallback should too:

```python
# 3-channel guided filter
for c in range(3):
    # Compute cross-covariance between guide channel and depth
    ...
# Or just use cv2.ximgproc.guidedFilter which already supports color
```

**Impact**: Low-medium. Most depth edges correlate with luminance, but some don't.

---

## 4. Strided Downsampling — Aliasing Problem

### Problem: Nearest-Neighbor Subsampling (gaussian_converter.py:65-66)

```python
colors = colors[stride//2::stride, stride//2::stride]
depth = depth[stride//2::stride, stride//2::stride]
```

Strided sampling with no anti-aliasing causes Moiré patterns and missed thin features. A stride-4 pass on a 4K panorama skips 15 out of every 16 pixels.

### Fix: Area-Average Downsampling

```python
# Anti-aliased downsampling with area averaging
if stride > 1:
    colors_4d = colors.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    colors_down = F.avg_pool2d(colors_4d, kernel_size=stride, stride=stride)
    colors = colors_down.squeeze(0).permute(1, 2, 0)
    
    depth_down = F.avg_pool2d(depth.unsqueeze(0).unsqueeze(0), stride, stride)
    depth = depth_down.squeeze()
```

For depth specifically, **min-pooling** might be even better to preserve foreground edges (closest depth wins):

```python
depth_down = -F.max_pool2d(-depth.unsqueeze(0).unsqueeze(0), stride, stride)
```

**Impact**: Medium, especially at stride ≥ 4. Eliminates aliasing artifacts visible as sparkle/noise in the splat output.

---

## 5. Spherical Harmonics — Currently Unused

### Problem: SH Degree 0 Only (ply_writer.py:111-114)

The code supports `sh_degree=3` in the PLY structure but fills all higher-order SH coefficients with zeros:

```python
for i in range(num_rest):
    data[f'f_rest_{i}'] = 0
```

This means every Gaussian is a flat-color disc. With SH degree 3, each Gaussian can represent view-dependent color variation (specular highlights, subsurface scattering approximation), dramatically improving realism.

### Fix: Compute SH Coefficients from Local Neighborhoods

For each Gaussian, sample the ERP image at several viewing angles around the Gaussian's position and fit SH basis functions:

```python
def compute_sh_coefficients(colors_grid, normals, degree=3):
    """Fit low-order SH to local color variation relative to viewing angle."""
    # For a single panorama, we only have one view, 
    # but we CAN compute angular color gradient from the ERP
    # which gives us SH band-1 coefficients (directional variation)
    ...
```

Even a simple approximation where band-1 SH encodes the local color gradient direction would add subtle view-dependent shading.

For multi-frame (video) input, you already have multiple viewpoints — the SH fitting becomes much more robust.

**Impact**: Medium-high for visual quality, especially in viewers that support SH rendering.

---

## 6. Sky Dome Quality

### Problem: Fixed Distance + Subsampled (gaussian_converter.py:196)

The sky dome uses `dome_distance=500.0`, `dome_stride=4`, and `dome_opacity=0.7`. This creates a visibly blocky, semi-transparent sky shell.

### Fix: Depth-Aware Sky Boundary + Hemisphere Sampling

Instead of a hard threshold, use a soft falloff:

```python
# Soft sky transition instead of hard threshold
sky_weight = torch.sigmoid((depth - sky_threshold) / (sky_threshold * 0.1))
# Gaussians near the threshold get blended opacity
dome_opacity_map = 0.7 + 0.25 * sky_weight
```

And use a Fibonacci sphere distribution for sky Gaussians instead of the ERP grid, which clusters samples at the poles:

```python
def fibonacci_sphere(n_points):
    """Generate evenly-spaced points on a hemisphere."""
    golden_ratio = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n_points)
    theta = 2 * math.pi * indices / golden_ratio
    phi = torch.acos(1 - 2 * indices / n_points)
    return theta, phi
```

**Impact**: Low-medium. Noticeable when looking "up" in the splat viewer.

---

## 7. SHARP Refinement Pipeline Issues

### Problem: Layer Averaging (sharp_refiner.py:371)

SHARP outputs 2 layers of Gaussians (front and back), but the code averages them:

```python
opacities = opacities.mean(dim=0)  # Average front+back layers
```

This loses the depth-stratification benefit that SHARP provides. The front layer captures surface detail; the back layer captures what's behind thin structures.

### Fix: Keep Both Layers

Use the front layer for surface Gaussians and the back layer only where the depth model indicates thin/semi-transparent geometry (foliage, fences, glass):

```python
# Use front layer primarily, back layer only for thin structures
front_opacity = opacities[0]
back_opacity = opacities[1]
# Identify thin structures by depth variance
thin_mask = depth_gradient_mag > threshold
combined = torch.where(thin_mask, 
                       (front_opacity + back_opacity) / 2,
                       front_opacity)
```

**Impact**: Medium. SHARP's multi-layer design is specifically for this, and collapsing it loses information.

### Problem: Scale Blend as Multiplier (gaussian_converter.py:423-428)

```python
scale_mult = ref_scales_flat / (ref_scales_flat.mean() + 1e-6)
scale_mult = scale_mult.clamp(0.5, 2.0)
```

This normalizes SHARP's scale predictions relative to their own mean, then uses them as a multiplier on the geometric scales. The clamping to [0.5, 2.0] limits SHARP's ability to correct genuinely oversized or undersized Gaussians.

### Fix: Use SHARP Scales More Directly

Let SHARP's absolute scale predictions carry more weight, especially at edges:

```python
# Compute a confidence weight from SHARP's opacity output
# High opacity areas → trust SHARP more
confidence = refined_attrs.opacities.clamp(0.1, 0.9)
adaptive_blend = scale_blend * confidence

base_gaussians['scales'] = (
    (1 - adaptive_blend) * base_gaussians['scales'] +
    adaptive_blend * ref_scales_projected
)
```

**Impact**: Medium. Allows SHARP to have more influence where it's confident.

---

## 8. Multi-Frame / Video Quality

### Problem: Frames Processed Independently

The video path (`convert_video` in cli.py) extracts frames and processes each one independently. There's no temporal consistency — the depth model may produce different depth scales between frames, causing the splat to "breathe" or shimmer.

### Fix: Temporal Depth Consistency

1. **Global scale alignment**: After estimating depth for all frames, compute a global scale that aligns overlapping regions:

```python
# Align consecutive frame depths using the visual odometry rotation
R_01 = estimate_rotation(frame0, frame1)
depth1_warped = warp_erp_depth(depth1, R_01)
# Find scale that minimizes ||depth0 - scale * depth1_warped|| in overlap
scale = (depth0 * depth1_warped).sum() / (depth1_warped ** 2).sum()
```

2. **Temporal smoothing**: Apply exponential moving average to depths:

```python
depth_smoothed = alpha * depth_current + (1 - alpha) * warp(depth_previous, R)
```

**Impact**: High for video input. Currently the biggest weakness of the video pipeline.

---

## 9. Color Space and Quantization

### Problem: uint8 Colors Only

Colors are stored as uint8 in SPLAT format and SH DC in PLY. There's no HDR support, and the SH DC encoding `(color - 0.5) / SH_C0` maps to a limited dynamic range.

### Fix: Scene-Adaptive Exposure + HDR-Aware Encoding

For HDR panorama inputs (common in 360° photography), preserve the dynamic range:

```python
# If input is HDR (float, values > 1.0), apply tone mapping for DC
# but preserve highlight info in SH band-1
if colors.max() > 1.0:
    # Reinhard tone mapping for DC
    dc_colors = colors / (1 + colors)
    # Store excess in SH band-1 as view-dependent highlight
    highlight = colors - dc_colors
    ...
```

**Impact**: Medium for HDR inputs, negligible for LDR.

---

## 10. Pole Handling — Currently Too Aggressive

### Problem: Hard Pole Exclusion (gaussian_converter.py:90-94)

```python
pole_mask[:pole_rows, :] = False
pole_mask[-pole_rows:, :] = False
```

This removes `pole_rows` (default 3) entire rows at both poles, creating visible holes at the zenith and nadir.

### Fix: Merge Pole Rows Instead of Removing

At the poles, the ERP grid converges — many pixels map to nearly the same 3D direction. Instead of removing them, merge them into fewer, larger Gaussians:

```python
# At poles, average every N pixels into single Gaussians
for row in range(pole_rows):
    merge_factor = max(2, W_grid // (4 * (row + 1)))
    # Pool colors and depth along azimuth
    row_colors = colors[row].reshape(-1, merge_factor, 3).mean(dim=1)
    row_depth = depth[row].reshape(-1, merge_factor).mean(dim=1)
    # Create larger Gaussians for these merged pixels
    ...
```

**Impact**: Low-medium. Only visible if you look straight up/down, but it's a visible hole in the current output.

---

## 11. Acceleration / Quality Tradeoffs

### Adaptive Stride

Instead of a global stride, use an **adaptive stride** based on depth:

```python
# Near objects → stride 1 (full detail)
# Far objects → stride 4 (acceptable since they're far away)
# This gives 4-16x more Gaussians where they matter most
stride_map = torch.where(depth < 5.0, 1,
             torch.where(depth < 20.0, 2, 4))
```

This would require reworking the grid-based approach to support variable density, but the quality/performance tradeoff is excellent.

### Gaussian Pruning Post-Pass

After generating all Gaussians, do a quick pruning pass:

```python
# Remove Gaussians that are fully occluded by closer ones
# Sort by distance from origin
distances = means.norm(dim=-1)
sorted_idx = distances.argsort()

# For each Gaussian, check if a closer one covers >90% of its footprint
# (This is a simplified version — full implementation uses splat rasterization)
```

**Impact**: Reduces file size 20-40% with minimal visual quality loss, which also improves viewer performance.

---

## Priority Summary

| # | Improvement | Impact | Effort | Where |
|---|------------|--------|--------|-------|
| 1 | Log/inverse depth mapping | 🔴 High | Low | `panda_model.py:270-281` |
| 2 | Depth-gradient adaptive opacity | 🟠 Med-High | Medium | `gaussian_converter.py:150` |
| 3 | Anti-aliased downsampling | 🟡 Medium | Low | `gaussian_converter.py:65-69` |
| 4 | Remove guided filter global rescaling | 🟡 Medium | Low | `depth_refiner.py:88-93` |
| 5 | Content-adaptive Gaussian scale | 🟡 Medium | Medium | `gaussian_converter.py:113-116` |
| 6 | SHARP dual-layer preservation | 🟡 Medium | Medium | `sharp_refiner.py:371` |
| 7 | SH degree 1+ coefficients | 🟠 Med-High | High | `ply_writer.py`, `gaussian_converter.py` |
| 8 | Higher-res PanDA inference | 🔴 High | Medium | `panda_model.py:217` |
| 9 | Temporal depth consistency (video) | 🔴 High | High | `cli.py`, new module |
| 10 | Pole merging instead of removal | 🟢 Low-Med | Medium | `gaussian_converter.py:90-94` |
| 11 | Adaptive stride by depth | 🟡 Medium | High | Architectural change |

The **top 3 quickest wins** that would noticeably improve output quality:

1. **Switch to log/inverse depth mapping** — ~10 lines changed, dramatic foreground improvement
2. **Remove the global rescaling in guided filter** — ~5 lines, fixes an active bug
3. **Use `F.avg_pool2d` instead of strided indexing** — ~8 lines, eliminates aliasing
