"""SPAG-4D cloud -> COLMAP scene bridge for the ArtiFixer3D backend.

Ports experiments/artifixer_eval/_bridge_to_colmap.py into a tested module.
ArtiFixer consumes a COLMAP scene (``images/`` + ``sparse/0/*.bin``) and rebuilds
its *own* 3DGRUT reconstruction — it does not ingest a PLY. This module renders an
orbit of perspective views from the SPAG cloud and writes that COLMAP scene, plus
the anchor/novel split the distill step requires.
"""
import struct
from pathlib import Path

import numpy as np

# COLMAP camera model ids — OPENCV = fx, fy, cx, cy, k1, k2, p1, p2
OPENCV_MODEL_ID = 4
SH_C0 = 0.28209479177387814


def select_anchor_indices(hole_fracs, quantile: float = 0.34):
    """Anchors = lowest-parallax (lowest hole-fraction) views.

    distill needs >=1 novel (non-anchor) view, so the anchor set is clamped
    to [1, n-1] views.
    """
    n = len(hole_fracs)
    order = sorted(range(n), key=lambda i: hole_fracs[i])
    k = max(1, min(n - 1, int(round(quantile * n))))
    return sorted(order[:k])


def rotmat2qvec(R):
    """World->cam rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    qvec = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_bin(path, width, height, fx, fy, cx, cy):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
        f.write(struct.pack("<iiQQ", 1, OPENCV_MODEL_ID, int(width), int(height)))
        f.write(struct.pack("<" + "d" * len(params), *params))


def write_images_bin(path, poses):
    """poses: list of (image_id, qvec(4), tvec(3), name)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(poses)))
        for img_id, qvec, tvec, name in poses:
            f.write(struct.pack("<idddddddi", img_id,
                                float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3]),
                                float(tvec[0]), float(tvec[1]), float(tvec[2]), 1))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # 0 2D observations (not needed for GS training)


def write_points3D_bin(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<Q", i + 1))                       # point3D id
            f.write(struct.pack("<ddd", *[float(v) for v in xyz[i]]))
            f.write(struct.pack("<BBB", int(rgb[i][0]), int(rgb[i][1]), int(rgb[i][2])))
            f.write(struct.pack("<d", 1.0))                         # reproj error
            f.write(struct.pack("<Q", 0))                           # empty track


def bridge_cloud_to_colmap(cloud_ply, out_dir, config):
    """Render an orbit of perspective views from a SPAG cloud and write a COLMAP scene.

    Refactor of experiments/artifixer_eval/_bridge_to_colmap.py:main() to take a
    config + paths. Needs the host GSFix3D CUDA rasterizer (heavy imports are local).

    Args:
        cloud_ply: path to the SPAG standard-3DGS PLY.
        out_dir: destination COLMAP scene dir (``images/``, ``masks/``, ``sparse/0/``).
        config: ArtiFixer3DConfig (num_directions, translation_fracs, fov_deg,
            render_resolution, num_seed_points).

    Returns:
        dict ``{"colmap_dir": str, "hole_fracs": [float], "n_views": int}``.
    """
    # Heavy / CUDA imports are local so the pure-Python writers + selector above
    # stay importable without torch or the rasterizer.
    import imageio.v3 as iio

    from .camera_rig import generate_camera_rig, render_with_hole_mask, _camera_to_RT
    from .format_compat import load_gaussians_from_ply

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    res = config.render_resolution
    g = load_gaussians_from_ply(str(cloud_ply), device="cuda")
    xyz = g.get_xyz.detach().cpu().numpy()

    # median radial depth from origin -> rig translation scale
    radial = np.linalg.norm(xyz, axis=1)
    median_depth = float(np.median(radial[radial > 0]))
    depth_proxy = np.full((16, 16), median_depth, dtype=np.float32)

    cams = generate_camera_rig(
        origin=np.zeros(3, dtype=np.float32), depth_map=depth_proxy,
        num_directions=config.num_directions, num_depths=len(config.translation_fracs),
        fov_deg=config.fov_deg, translation_fracs=tuple(config.translation_fracs),
        resolution=res,
    )

    K = cams[0].intrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    write_cameras_bin(out_dir / "sparse/0/cameras.bin", res, res, fx, fy, cx, cy)

    poses = []
    hole_fracs = []
    for i, cam in enumerate(cams):
        rgb, hole = render_with_hole_mask(g, cam, alpha_threshold=0.05)
        name = f"frame_{i:05d}.png"
        iio.imwrite(out_dir / "images" / name, (np.clip(rgb, 0, 1) * 255).astype("uint8"))
        iio.imwrite(out_dir / "masks" / name, ((1 - hole) * 255).astype("uint8"))
        R, T = _camera_to_RT(cam)
        poses.append((i + 1, rotmat2qvec(R), T, name))
        hole_fracs.append(float(hole.mean()))
    write_images_bin(out_dir / "sparse/0/images.bin", poses)

    # seed points3D from subsampled Gaussian centers + DC colour
    n = len(xyz)
    idx = np.random.default_rng(0).choice(n, size=min(config.num_seed_points, n), replace=False)
    try:
        dc = g._features_dc.detach().cpu().numpy().reshape(n, 3)[idx]
        rgb_pts = np.clip(SH_C0 * dc + 0.5, 0, 1) * 255
    except Exception:
        rgb_pts = np.full((len(idx), 3), 128.0)
    write_points3D_bin(out_dir / "sparse/0/points3D.bin", xyz[idx], rgb_pts.astype(int))

    return {"colmap_dir": str(out_dir), "hole_fracs": hole_fracs, "n_views": len(poses)}
