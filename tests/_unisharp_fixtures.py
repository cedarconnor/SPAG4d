"""Helpers that synthesize a UniSHARP-style 3DGS PLY for tests.

The real UniSHARP writer emits a standard INRIA vertex element plus extra
"supplement" elements (extrinsic / intrinsic / image_size) and a color_space
marker. PLY has no native string property, so we encode color_space as a PLY
comment here; the real encoding is verified during the M1.5 viewer check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

CORE_VERTEX_FIELDS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
    + ["opacity"]
)


def write_fake_unisharp_ply(
    path: str | Path,
    n_vertices: int = 25,
    with_supplements: bool = True,
    color_space: str | None = "sRGB",
) -> Path:
    """Write a UniSHARP-shaped PLY and return its path."""
    path = Path(path)
    rng = np.random.RandomState(0)  # deterministic; no Math.random/Date needed

    dtype = [(name, "f4") for name in CORE_VERTEX_FIELDS]
    vtx = np.empty(n_vertices, dtype=dtype)
    for name in CORE_VERTEX_FIELDS:
        vtx[name] = rng.rand(n_vertices).astype("f4")
    # Make quaternions look normalized-ish so the file is plausible.
    vtx["rot_0"] = 1.0

    elements = [PlyElement.describe(vtx, "vertex")]

    if with_supplements:
        extr = np.empty(4, dtype=[(f"m{i}", "f4") for i in range(4)])
        for i in range(4):
            extr[f"m{i}"] = np.eye(4, dtype="f4")[:, i]
        intr = np.empty(3, dtype=[(f"m{i}", "f4") for i in range(3)])
        for i in range(3):
            intr[f"m{i}"] = np.eye(3, dtype="f4")[:, i]
        size = np.array([(1536, 768)], dtype=[("width", "i4"), ("height", "i4")])
        elements.append(PlyElement.describe(extr, "extrinsic"))
        elements.append(PlyElement.describe(intr, "intrinsic"))
        elements.append(PlyElement.describe(size, "image_size"))

    comments = []
    if color_space is not None:
        comments.append(f"color_space {color_space}")

    PlyData(elements, text=False, comments=comments).write(str(path))
    return path
