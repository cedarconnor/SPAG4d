# spag4d/refine/geometric/depth_align.py
"""Robust IRLS scale/shift alignment of per-frame depth to rendered base splat depth."""
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class AlignmentResult:
    scale: float
    shift: float
    inlier_count: int
    inlier_fraction: float
    residual_median: float
    residual_p95: float
    converged: bool


def align_depth_irls(
    depth_raw: np.ndarray,
    depth_rendered: np.ndarray,
    mask: np.ndarray,
    mode: Literal["scale_shift", "scale_only"] = "scale_shift",
    max_iters: int = 10,
    min_inlier_fraction: float = 0.20,
    convergence_tol: float = 1e-4,
) -> AlignmentResult:
    d = depth_raw[mask].astype(np.float64)
    r = depth_rendered[mask].astype(np.float64)
    n = len(d)

    if n < 10:
        return AlignmentResult(1.0, 0.0, 0, 0.0, np.inf, np.inf, False)

    huber_delta = 0.05 * np.median(r)
    s, t = 1.0, 0.0
    prev_s, prev_t = 0.0, 0.0

    for _ in range(max_iters):
        residuals = s * d + t - r
        abs_res = np.abs(residuals)
        weights = np.where(abs_res <= huber_delta, 1.0, huber_delta / (abs_res + 1e-10))

        if mode == "scale_shift":
            A = np.column_stack([d, np.ones(n)])
            AtWA = A.T @ (weights[:, None] * A)
            AtWb = A.T @ (weights * r)
            try:
                params = np.linalg.solve(AtWA, AtWb)
            except np.linalg.LinAlgError:
                break
            s, t = float(params[0]), float(params[1])
        else:
            s = float(np.sum(weights * r * d) / (np.sum(weights * d * d) + 1e-10))
            t = 0.0

        if abs(s - prev_s) < convergence_tol and abs(t - prev_t) < convergence_tol:
            break
        prev_s, prev_t = s, t

    final_residuals = np.abs(s * d + t - r)
    median_depth = np.median(r) + 1e-8
    inlier_mask = final_residuals < (0.25 * median_depth)
    inlier_count = int(inlier_mask.sum())
    inlier_fraction = inlier_count / n

    res_median = float(np.median(final_residuals))
    # Compute p95 on inliers only so outlier corruption doesn't poison the convergence check
    inlier_residuals = final_residuals[inlier_mask] if inlier_count > 0 else final_residuals
    res_p95 = float(np.percentile(inlier_residuals, 95))

    converged = (
        inlier_fraction >= min_inlier_fraction
        and (res_p95 / median_depth) < 0.25
    )

    return AlignmentResult(
        scale=s,
        shift=t,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        residual_median=res_median,
        residual_p95=res_p95,
        converged=converged,
    )
