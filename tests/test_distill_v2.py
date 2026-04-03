# tests/test_distill_v2.py
import numpy as np
import pytest
import torch

from spag4d.refine.distill import compute_weighted_loss


class TestComputeWeightedLoss:
    def test_tier1_full_weight(self):
        rendered = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        loss = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=None)
        assert loss.item() > 0

    def test_tier2_reduced_weight(self):
        rendered = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        loss_full = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=None)
        loss_low = compute_weighted_loss(rendered, gt, tier2_weight=0.2, hole_mask=None)
        assert loss_low.item() < loss_full.item()
        assert loss_low.item() == pytest.approx(loss_full.item() * 0.2, rel=0.01)

    def test_hole_mask_excludes_observed(self):
        rendered = torch.zeros(3, 64, 64)
        gt = torch.ones(3, 64, 64)
        mask = torch.zeros(64, 64)
        mask[:32, :32] = 1.0
        loss_masked = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=mask)
        loss_full = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=None)
        assert loss_masked.item() > 0
        assert loss_masked.item() <= loss_full.item()

    def test_empty_mask_produces_zero(self):
        rendered = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        mask = torch.zeros(64, 64)
        loss = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)
