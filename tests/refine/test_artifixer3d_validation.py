import numpy as np

from spag4d.refine.artifixer3d_pipeline import _artifixer_metrics


def test_metrics_finite_and_in_range():
    # one "anchor" view: refined matches original closely; refined has fewer holes
    orig_rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    refined_rgb = orig_rgb.copy()
    orig_hole = np.zeros((8, 8), dtype=np.float32)
    orig_hole[:, :4] = 1.0        # 50% holes
    refined_hole = np.zeros((8, 8), dtype=np.float32)
    refined_hole[:, :1] = 1.0     # 12.5% holes
    m = _artifixer_metrics([orig_rgb], [refined_rgb], [orig_hole], [refined_hole])
    assert np.isfinite(m["anchor_psnr"])
    assert 0.0 <= m["coverage_before"] <= 1.0
    assert 0.0 <= m["coverage_after"] <= 1.0
    # ArtiFixer must not reduce anchor coverage
    assert m["coverage_after"] >= m["coverage_before"]


def test_identical_renders_give_high_psnr():
    img = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    z = np.zeros((4, 4), dtype=np.float32)
    m = _artifixer_metrics([img], [img.copy()], [z], [z])
    assert m["anchor_psnr"] >= 99.0   # effectively identical
