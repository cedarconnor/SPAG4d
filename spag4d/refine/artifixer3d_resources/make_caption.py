"""Generate the caption HDF5 WITHOUT the 60GB Qwen3-VL-30B VLM (won't fit 48GB usefully).
Monkeypatch the VLM to return a fixed caption, then reuse ArtiFixer's real
generate_caption_hdf5 (exact image-loading + HDF5 format) + UMT5 text encoder.

Vendored from experiments/artifixer_eval/_make_caption.py (Phase 0/1 validated).
Scene paths are parameterized via env vars so the adapter can target any scene:
    SCENE_ROOT  -- e.g. /scene/prep/bell_tower  (the per-scene prep root)
    SCENE_NAME  -- e.g. bell_tower              (the scene/dataset name)
    SCENE_CAPTION (optional) -- override the fixed caption text
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/artifixer")
import data_processing.captioning.generate_captions as gc

SCENE_ROOT = os.environ["SCENE_ROOT"]
SCENE_NAME = os.environ.get("SCENE_NAME", "bell_tower")

# Manual scene caption. Quality is secondary — only the UMT5 embedding is consumed.
CAPTION = os.environ.get(
    "SCENE_CAPTION",
    "A stone chapel interior with tall pointed arched windows of stained glass, "
    "bell ropes hanging from the high ceiling, bare stone walls and a stone floor, "
    "soft natural daylight filling the space.",
)


class _Dummy:
    @staticmethod
    def from_pretrained(*a, **k):
        return None


# Stub the captioning VLM so no 60GB download/load happens.
gc.Qwen3VLMoeForConditionalGeneration = _Dummy
gc.Qwen3VLProcessor = _Dummy
gc.generate_caption = lambda *a, **k: CAPTION

gc.generate_caption_hdf5(
    input_path=Path(SCENE_ROOT) / "3dgrut_input" / SCENE_NAME,
    output_path=Path(SCENE_ROOT) / "captions" / SCENE_NAME / "caption.h5",
    dataset_downsample_factor=1,
    # Local folder fetched via wget -c (avoids HF xet hang / snapshot_download exit-255).
    # UMT5-XXL text_encoder + tokenizer; weights identical across Wan2.1 1.3B/14B repos.
    text_encoder_model_id="/data/wan_te",
)
print("CAPTION HDF5 DONE")
