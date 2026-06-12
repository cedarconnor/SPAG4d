"""UniSHARP PLY format handling: inspect, count, copy, convert."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)

CORE_VERTEX_FIELDS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
    + ["opacity"]
)


def _read_color_space(ply) -> str | None:
    """Best-effort color_space read. Tries PLY comments first.

    The real UniSHARP encoding is unverified (PLY has no string property type),
    so this is tolerant: it returns None rather than raising when absent.
    """
    for c in getattr(ply, "comments", []) or []:
        parts = c.strip().split()
        if len(parts) == 2 and parts[0] == "color_space":
            return parts[1]
    return None


def inspect_ply_fields(ply_path: str) -> dict:
    """Report element names, vertex field names, and supplement elements."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    elements = {el.name: [p.name for p in el.properties] for el in ply.elements}
    vertex_fields = elements.get("vertex", [])
    supplements = [name for name in elements if name != "vertex"]
    return {
        "elements": elements,
        "vertex_fields": vertex_fields,
        "supplement_elements": supplements,
        "has_core_fields": all(f in vertex_fields for f in CORE_VERTEX_FIELDS),
        "color_space": _read_color_space(ply),
    }


def count_ply_vertices(ply_path: str) -> int:
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    for el in ply.elements:
        if el.name == "vertex":
            return int(el.count)
    return 0


def copy_unisharp_ply_to_output(src_path: str, dst_path: str) -> dict:
    """Copy raw UniSHARP PLY verbatim and count vertices.

    Keeps supplement elements intact. Works in viewers that tolerate extra PLY
    elements (SuperSplat, GaussianSplats3D). Use convert mode for strict loaders.
    """
    shutil.copy2(src_path, dst_path)
    n = count_ply_vertices(dst_path)
    info = inspect_ply_fields(dst_path)
    if info["supplement_elements"]:
        LOGGER.info(
            "UniSHARP PLY carries supplement elements %s; "
            "strict loaders may need format_mode='convert'.",
            info["supplement_elements"],
        )
    return {"num_gaussians": n, "ply_info": info}


def convert_unisharp_ply_to_spag(src_path: str, dst_path: str) -> dict:
    """Rewrite UniSHARP PLY as a clean INRIA 3DGS PLY.

    Drops supplement elements, keeps exactly CORE_VERTEX_FIELDS in order.
    Scales stay log, opacity stays logit, quats stay wxyz. Colors (f_dc) are
    preserved as-is by default; color_space conversion is deferred until the
    real encoding is confirmed (M1.5).
    """
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(src_path)
    vtx = next(el for el in ply.elements if el.name == "vertex")
    present = {p.name for p in vtx.properties}

    missing = [f for f in CORE_VERTEX_FIELDS if f not in present]
    if missing:
        raise KeyError(f"UniSHARP PLY missing core fields: {missing}")

    dtype = [(name, "f4") for name in CORE_VERTEX_FIELDS]
    arr = np.empty(int(vtx.count), dtype=dtype)
    for name in CORE_VERTEX_FIELDS:
        arr[name] = np.asarray(vtx[name]).astype("f4")

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(dst_path)
    return {"num_gaussians": int(len(arr)), "converted": True}
