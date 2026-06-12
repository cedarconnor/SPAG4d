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


def reorient_gaussians_inplace(arr) -> None:
    """Rotate a 3DGS vertex array 180 degrees about the X axis, in place.

    UniSHARP/UniK3D emit gaussians in a camera-at-origin frame with +Y pointing
    DOWN (OpenCV-style), so the splat loads upside-down in SPAG4d's +Y-up viewer.
    A 180 deg rotation about X is the proper-rotation fix: positions
    (x, y, z) -> (x, -y, -z); it also flips Z (forward<->back), which is
    cosmetically irrelevant for a full 360 panorama.

    The matching quaternion (wxyz) update for q_R = (0, 1, 0, 0):
        (w, x, y, z) -> (-x, w, -z, y)

    `arr` must be a structured numpy array with fields x, y, z and rot_0..rot_3.
    """
    arr["y"] = -arr["y"]
    arr["z"] = -arr["z"]
    w = arr["rot_0"].copy()
    rx = arr["rot_1"].copy()
    ry = arr["rot_2"].copy()
    rz = arr["rot_3"].copy()
    arr["rot_0"] = -rx
    arr["rot_1"] = w
    arr["rot_2"] = -rz
    arr["rot_3"] = ry


def denoise_mask(arr, opacity_min: float = 0.05, sor_k: int = 16,
                 sor_std_ratio: float = 2.0):
    """Boolean keep-mask that removes faint and isolated gaussians.

    UniSHARP's monocular output leaves low-opacity speckle and stretched
    floaters around depth discontinuities (e.g. car edges) that SPAG4d's
    depth-based generators prune but the raw UniSHARP path does not. This pass:

      1. drops gaussians whose opacity (sigmoid of the stored logit) is below
         `opacity_min`;
      2. statistical-outlier-removal: drops points whose mean distance to their
         `sor_k` nearest neighbours exceeds mean + `sor_std_ratio` * std.

    Returns a boolean numpy array of length len(arr).
    """
    import numpy as np

    n = len(arr)
    keep = np.ones(n, dtype=bool)

    if opacity_min and opacity_min > 0.0:
        opacity = 1.0 / (1.0 + np.exp(-arr["opacity"].astype("f8")))
        keep &= opacity >= float(opacity_min)

    if sor_k and sor_std_ratio and int(keep.sum()) > sor_k + 1:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            LOGGER.warning("scipy not installed; skipping outlier removal.")
            return keep
        idx = np.where(keep)[0]
        xyz = np.stack([arr["x"][idx], arr["y"][idx], arr["z"][idx]], axis=1).astype("f8")
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=sor_k + 1, workers=-1)
        mean_d = dists[:, 1:].mean(axis=1)
        thresh = float(mean_d.mean() + sor_std_ratio * mean_d.std())
        sub_keep = mean_d <= thresh
        keep[idx[~sub_keep]] = False

    return keep


def convert_unisharp_ply_to_spag(
    src_path: str,
    dst_path: str,
    reorient: bool = True,
    denoise: bool = True,
    opacity_min: float = 0.05,
    sor_k: int = 16,
    sor_std_ratio: float = 2.0,
) -> dict:
    """Rewrite UniSHARP PLY as a clean, corrected INRIA 3DGS PLY.

    Drops supplement elements, keeps exactly CORE_VERTEX_FIELDS in order.
    Scales stay log, opacity stays logit, quats stay wxyz. Colors (f_dc) are
    preserved as-is. Additionally (production defaults):
      - `reorient`: fix UniSHARP's upside-down frame (see reorient_gaussians_inplace);
      - `denoise`: prune faint/isolated gaussians (see denoise_mask).
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

    if reorient:
        reorient_gaussians_inplace(arr)

    n_in = len(arr)
    if denoise:
        mask = denoise_mask(arr, opacity_min=opacity_min, sor_k=sor_k,
                            sor_std_ratio=sor_std_ratio)
        arr = arr[mask]
        removed = n_in - len(arr)
        if removed > 0:
            LOGGER.info("UniSHARP denoise removed %d/%d gaussians.", removed, n_in)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(dst_path)
    return {"num_gaussians": int(len(arr)), "converted": True}
