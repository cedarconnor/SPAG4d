"""Export a distilled ArtiFixer3D 3DGRUT checkpoint to a standard 3DGS PLY,
via 3DGRUT's native exporter (the bridge-back: 3DGRUT ckpt -> SPAG-loadable PLY).

Vendored from experiments/artifixer_eval/_export_ply.py (Phase 1 validated).
Paths are taken from argv so the adapter can point at any distill checkpoint:
    python export_ply.py <ckpt_distill.pt> <out.ply>
"""
import os
import sys

import torch

sys.path.insert(0, "/workspace/artifixer")
sys.path.insert(0, "/workspace/thirdparty/3DGRUT-ArtiFixer")
from threedgrut.model.model import MixtureOfGaussians

CKPT = sys.argv[1]
OUT = sys.argv[2]

ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
conf = ckpt["config"]
print("loaded checkpoint, global_step=", ckpt.get("global_step"), "N=", ckpt["positions"].shape[0])
model = MixtureOfGaussians(conf, scene_extent=ckpt.get("scene_extent"))
model.init_from_checkpoint(ckpt, setup_optimizer=False)
model.export_ply(OUT)
print("EXPORTED_PLY:", OUT)
print("ply size bytes:", os.path.getsize(OUT))
