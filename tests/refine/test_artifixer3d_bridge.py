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


def test_write_cameras_bin_roundtrip(tmp_path):
    from spag4d.refine.artifixer3d_bridge import write_cameras_bin
    p = tmp_path / "cameras.bin"
    write_cameras_bin(p, width=640, height=640, fx=500.0, fy=500.0, cx=320.0, cy=320.0)
    data = p.read_bytes()
    (n_cams,) = struct.unpack("<Q", data[:8])
    assert n_cams == 1
    cam_id, model_id, w, h = struct.unpack("<iiQQ", data[8:28])
    assert model_id == 4 and w == 640 and h == 640


def test_write_images_bin_count(tmp_path):
    from spag4d.refine.artifixer3d_bridge import write_images_bin
    p = tmp_path / "images.bin"
    poses = [(1, np.array([1.0, 0, 0, 0]), np.array([0.1, 0.2, 0.3]), "frame_00000.png")]
    write_images_bin(p, poses)
    (n,) = struct.unpack("<Q", p.read_bytes()[:8])
    assert n == 1
