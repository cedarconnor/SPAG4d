import struct

import numpy as np

from spag4d.refine.artifixer3d_bridge import select_anchor_indices


def test_low_parallax_views_are_anchors():
    # 6 views: parallax (proxy = hole fraction) ascending
    hole_fracs = [0.02, 0.05, 0.10, 0.40, 0.55, 0.70]
    anchors = select_anchor_indices(hole_fracs, quantile=0.34)
    # lowest-third are anchors; the rest are novel; at least one of each
    assert set(anchors) == {0, 1}
    novel = [i for i in range(6) if i not in anchors]
    assert novel and anchors


def test_never_returns_all_or_none():
    anchors = select_anchor_indices([0.1] * 10, quantile=0.34)
    assert 0 < len(anchors) < 10   # distill requires >=1 novel AND anchors exist
