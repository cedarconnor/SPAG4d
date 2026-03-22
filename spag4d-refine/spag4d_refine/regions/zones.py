"""RepairZone: connected component grouping of repair regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.ndimage import label as connected_components

from .classifier import RegionType


@dataclass
class RepairZone:
    """A connected region requiring repair in a single frame."""
    frame_idx: int
    zone_id: int
    region_type: int          # RegionType value
    mask: np.ndarray          # [H, W] bool
    bbox: tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)
    area: int                 # Number of pixels

    @property
    def centroid(self) -> tuple[float, float]:
        """(y, x) centroid of the zone."""
        ys, xs = np.where(self.mask)
        return (float(np.mean(ys)), float(np.mean(xs)))


def extract_repair_zones(
    region_map: np.ndarray,
    frame_idx: int = 0,
    min_area: int = 50,
    target_types: Optional[List[int]] = None,
) -> List[RepairZone]:
    """
    Extract connected repair zones from a region classification map.

    Args:
        region_map: [H, W] int with RegionType values
        frame_idx: Index of this frame in the trajectory
        min_area: Minimum zone area in pixels (skip tiny artifacts)
        target_types: Which RegionTypes to extract (default: TYPE_A, TYPE_B, TYPE_C)

    Returns:
        List of RepairZone objects, sorted by area (largest first)
    """
    if target_types is None:
        target_types = [RegionType.TYPE_A, RegionType.TYPE_B, RegionType.TYPE_C]

    zones = []
    zone_id = 0

    for rtype in target_types:
        type_mask = region_map == rtype
        if not type_mask.any():
            continue

        labeled, n_components = connected_components(type_mask)

        for comp in range(1, n_components + 1):
            comp_mask = labeled == comp
            area = int(np.sum(comp_mask))
            if area < min_area:
                continue

            ys, xs = np.where(comp_mask)
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

            zones.append(RepairZone(
                frame_idx=frame_idx,
                zone_id=zone_id,
                region_type=rtype,
                mask=comp_mask,
                bbox=bbox,
                area=area,
            ))
            zone_id += 1

    # Sort by area, largest first
    zones.sort(key=lambda z: z.area, reverse=True)
    return zones
