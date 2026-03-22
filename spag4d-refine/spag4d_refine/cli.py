"""CLI for spag4d-refine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger("spag4d_refine")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """SPAG-4D Refinement: repair Gaussian splat gaps with AI synthesis."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.argument("splat_path", type=click.Path(exists=True))
@click.argument("panorama_path", type=click.Path(exists=True))
@click.option("--panorama-depth", type=click.Path(exists=True), help="Panoramic depth map (.npy)")
@click.option("--cameras", type=click.Path(exists=True), help="Camera set JSON file")
@click.option("--output-dir", "-o", type=click.Path(), default="./refine_output")
@click.option("--backend", type=click.Choice(["klein-sharp", "flux-fill", "sdxl"]), default="klein-sharp")
@click.option("--rounds", type=int, default=2, help="Max refinement rounds")
@click.option("--warp-subsample", type=int, default=1, help="Forward warp subsampling factor (1=full, 2=half, etc.)")
@click.option("--device", type=str, default="cuda")
def refine(
    splat_path: str,
    panorama_path: str,
    panorama_depth: Optional[str],
    cameras: Optional[str],
    output_dir: str,
    backend: str,
    rounds: int,
    warp_subsample: int,
    device: str,
) -> None:
    """Run the full refinement pipeline on a Gaussian splat PLY file."""
    from .config import RefineConfig
    from .pipeline import RefinePipeline
    from .camera.pinhole import CameraSet

    config = RefineConfig(
        splat_path=Path(splat_path),
        panorama_rgb_path=Path(panorama_path),
        panorama_depth_path=Path(panorama_depth) if panorama_depth else None,
        output_dir=Path(output_dir),
        synthesis_backend=backend,
        max_refinement_rounds=rounds,
        warp_subsample=warp_subsample,
        device=device,
    )

    camera_set = None
    if cameras:
        camera_set = CameraSet.load_json(cameras)

    pipeline = RefinePipeline(config)
    result = pipeline.run(cameras=camera_set)

    click.echo(f"\nRefinement complete:")
    click.echo(f"  Output: {result.output_path}")
    click.echo(f"  Gaussians: {result.original_gaussian_count:,} → {result.final_gaussian_count:,}")
    click.echo(f"  Rounds: {result.rounds_completed}")
    click.echo(f"  Time: {result.total_elapsed_seconds:.1f}s")


@main.command()
@click.argument("splat_path", type=click.Path(exists=True))
@click.argument("panorama_path", type=click.Path(exists=True))
@click.option("--cameras", type=click.Path(exists=True), required=True, help="Camera set JSON")
@click.option("--output-dir", "-o", type=click.Path(), default="./extract_output")
@click.option("--panorama-depth", type=click.Path(exists=True))
def extract(
    splat_path: str,
    panorama_path: str,
    cameras: str,
    output_dir: str,
    panorama_depth: Optional[str],
) -> None:
    """Extract panoramic views and forward warps (stages 0-2b only)."""
    import numpy as np
    from PIL import Image
    from .camera.pinhole import CameraSet
    from .camera.panoramic_extractor import extract_panoramic_view

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    camera_set = CameraSet.load_json(cameras)
    panorama_rgb = np.array(Image.open(panorama_path).convert("RGB")).astype(np.float32) / 255.0

    panorama_depth_arr = None
    if panorama_depth:
        panorama_depth_arr = np.load(panorama_depth)

    for i, cam in enumerate(camera_set):
        # Panoramic extraction
        pano_result = extract_panoramic_view(panorama_rgb, panorama_depth_arr or np.ones(panorama_rgb.shape[:2]), cam)
        img = Image.fromarray((np.clip(pano_result.rgb, 0, 1) * 255).astype(np.uint8))
        img.save(out / f"pano_extract_{i:03d}.png")

        # Forward warp
        if panorama_depth_arr is not None:
            from .camera.forward_warper import forward_warp_panorama
            warp = forward_warp_panorama(panorama_rgb, panorama_depth_arr, cam)
            img = Image.fromarray((np.clip(warp.warped_rgb, 0, 1) * 255).astype(np.uint8))
            img.save(out / f"forward_warp_{i:03d}.png")

            from .synthesis.depth_visualizer import depth_to_disparity_image
            depth_vis = depth_to_disparity_image(warp.warped_depth, warp.valid_mask)
            img = Image.fromarray((np.clip(depth_vis, 0, 1) * 255).astype(np.uint8))
            img.save(out / f"depth_vis_{i:03d}.png")

    click.echo(f"Extracted {len(camera_set)} views to {out}")


@main.command()
@click.argument("splat_path", type=click.Path(exists=True))
@click.option("--cameras", type=click.Path(exists=True), required=True, help="Camera set JSON")
@click.option("--output-dir", "-o", type=click.Path(), default="./render_output")
@click.option("--device", type=str, default="cuda")
def render(
    splat_path: str,
    cameras: str,
    output_dir: str,
    device: str,
) -> None:
    """Render a Gaussian splat from camera positions (stage 3 only)."""
    import numpy as np
    from PIL import Image
    from .gaussian.cloud import GaussianCloud
    from .camera.pinhole import CameraSet
    from .renderer.gsplat_renderer import GsplatRenderer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cloud = GaussianCloud.from_ply(splat_path)
    camera_set = CameraSet.load_json(cameras)
    renderer = GsplatRenderer(cloud, device=device)

    for i, cam in enumerate(camera_set):
        result = renderer.render(cam)
        img = Image.fromarray((np.clip(result.rgb, 0, 1) * 255).astype(np.uint8))
        img.save(out / f"render_{i:03d}.png")

    click.echo(f"Rendered {len(camera_set)} views to {out}")


@main.command()
@click.argument("splat_path", type=click.Path(exists=True))
@click.argument("panorama_path", type=click.Path(exists=True))
@click.option("--cameras", type=click.Path(exists=True), required=True, help="Camera set JSON")
@click.option("--panorama-depth", type=click.Path(exists=True), required=True)
@click.option("--output-dir", "-o", type=click.Path(), default="./classify_output")
@click.option("--device", type=str, default="cuda")
def classify(
    splat_path: str,
    panorama_path: str,
    cameras: str,
    panorama_depth: str,
    output_dir: str,
    device: str,
) -> None:
    """Run region classification on a splat + panorama (stages 0-4)."""
    import numpy as np
    from PIL import Image
    from .gaussian.cloud import GaussianCloud
    from .camera.pinhole import CameraSet
    from .camera.forward_warper import forward_warp_panorama
    from .renderer.gsplat_renderer import GsplatRenderer
    from .renderer.diagnostics import save_diagnostic_bundle
    from .regions.classifier import classify_frame

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cloud = GaussianCloud.from_ply(splat_path)
    camera_set = CameraSet.load_json(cameras)
    panorama_rgb = np.array(Image.open(panorama_path).convert("RGB")).astype(np.float32) / 255.0
    depth = np.load(panorama_depth)

    renderer = GsplatRenderer(cloud, device=device)

    for i, cam in enumerate(camera_set):
        warp = forward_warp_panorama(panorama_rgb, depth, cam)
        render_result = renderer.render(cam)

        region_map = classify_frame(
            warp_rgb=warp.warped_rgb,
            warp_valid=warp.valid_mask,
            splat_rgb=render_result.rgb,
            splat_alpha=render_result.alpha,
        )

        save_diagnostic_bundle(
            splat_rgb=render_result.rgb,
            warp_rgb=warp.warped_rgb,
            pano_rgb=None,
            region_map=region_map,
            output_path=out / f"diagnostic_{i:03d}.png",
        )

    click.echo(f"Classified {len(camera_set)} views to {out}")


if __name__ == "__main__":
    main()
