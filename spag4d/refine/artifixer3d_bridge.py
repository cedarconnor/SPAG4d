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
