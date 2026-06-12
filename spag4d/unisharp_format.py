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
