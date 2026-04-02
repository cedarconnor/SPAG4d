"""Gaussian provenance tracking: original vs. refinement-created."""
import logging

logger = logging.getLogger(__name__)


def tag_gaussian_provenance(gaussians, initial_count):
    logger.info(f"[stub] tag_gaussian_provenance(initial={initial_count})")


def apply_provenance_lr_scaling(gaussians, optimizer, initial_count, scale=0.1):
    logger.info(f"[stub] apply_provenance_lr_scaling(scale={scale})")
