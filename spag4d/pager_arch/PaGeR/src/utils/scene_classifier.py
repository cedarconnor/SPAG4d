"""CLIP ViT-B/32 indoor/outdoor classifier for routing PaGeR's twin scale heads.

Operates on the 4 equatorial cubemap faces already produced by the depth pipeline.
"""

from __future__ import annotations

from typing import Sequence, Union

import open_clip
import torch
import torch.nn.functional as F


DEFAULT_INDOOR_PROMPTS: tuple[str, ...] = (
    "an indoor scene",
    "the interior of a building",
    "a room inside a building",
    "a hallway",
    "an office room",
    "a bedroom",
    "a living room",
    "a kitchen",
    "an indoor space with walls and a ceiling",
)
DEFAULT_OUTDOOR_PROMPTS: tuple[str, ...] = (
    "an outdoor scene",
    "the outdoors",
    "outside in nature",
    "a street view",
    "a city street",
    "an urban panorama",
    "a park",
    "a landscape",
    "a residential neighborhood",
)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class IndoorOutdoorClassifier:
    """CLIP-based indoor/outdoor classifier; class centroids = L2-normalised mean of prompt embeddings."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        indoor_prompts: Sequence[str] = DEFAULT_INDOOR_PROMPTS,
        outdoor_prompts: Sequence[str] = DEFAULT_OUTDOOR_PROMPTS,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # force_quick_gelu matches OpenAI's training activation; otherwise open_clip silently falls back to GELU.
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
            force_quick_gelu=(pretrained == "openai"),
        )
        self.model = model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.inference_mode():
            centroids = {}
            for key, prompts in (("indoor", indoor_prompts), ("outdoor", outdoor_prompts)):
                toks = tokenizer(list(prompts)).to(self.device)
                feats = F.normalize(self.model.encode_text(toks), dim=-1)
                centroids[key] = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
        self.text_indoor = centroids["indoor"]
        self.text_outdoor = centroids["outdoor"]

        self.register_clip_norm(self.device)

    def register_clip_norm(self, device: torch.device) -> None:
        self._clip_mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self._clip_std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def p_outdoor(self, cubemap_01: torch.Tensor) -> float:
        """P(outdoor) averaged over the 4 equatorial faces of a (6, 3, F, F) cubemap in [0, 1]."""
        if cubemap_01.ndim == 5:
            cubemap_01 = cubemap_01[0]
        if cubemap_01.ndim != 4 or cubemap_01.shape[0] < 4:
            raise ValueError(
                f"Expected cubemap of shape (6, 3, F, F); got {tuple(cubemap_01.shape)}"
            )
        eq = cubemap_01[:4].to(self.device).clamp(0, 1)
        eq = F.interpolate(eq, size=(224, 224), mode="bilinear",
                           align_corners=False, antialias=True)
        eq = (eq - self._clip_mean) / self._clip_std
        feats = F.normalize(self.model.encode_image(eq), dim=-1)
        s_in = (feats @ self.text_indoor.T).squeeze(-1)
        s_out = (feats @ self.text_outdoor.T).squeeze(-1)
        probs = torch.stack([s_in, s_out], dim=-1).mul(100.0).softmax(dim=-1)
        return float(probs[:, 1].mean().item())

    def classify(self, cubemap_01: torch.Tensor, threshold: float = 0.5):
        p = self.p_outdoor(cubemap_01)
        return ("outdoor" if p >= threshold else "indoor", p)


_SINGLETON: "IndoorOutdoorClassifier | None" = None


def get_classifier(device: Union[str, torch.device] = "cuda") -> IndoorOutdoorClassifier:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = IndoorOutdoorClassifier(device=device)
    return _SINGLETON
