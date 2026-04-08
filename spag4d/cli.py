# spag4d/cli.py
"""
Command-line interface for SPAG-4D.
"""

import click
from pathlib import Path


@click.group()
@click.version_option(version="3.0.0")
def main():
    """SPAG-4D: Convert 360° panoramas to 3D Gaussian Splats."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--depth-model', type=click.Choice(['dap', 'da360']),
              default='da360', help='Depth estimation model (default: da360)')
@click.option('--sharp-refine', is_flag=True,
              help='Experimental: SHARP per-face refinement (slower, may not improve quality)')
@click.option('--stride', type=int, default=2,
              help='SPAG pixel stride: 1=full, 2=quarter, 4=sixteenth (SPAG mode only)')
@click.option('--depth-min', default=0.1, help='Minimum depth in meters')
@click.option('--depth-max', default=100.0, help='Maximum depth in meters')
@click.option('--sky-threshold', default=80.0, help='Sky depth threshold (0 to disable)')
@click.option('--outlier-pruning', default=0.0, help='Outlier removal strength (0=off, 1=aggressive)')
@click.option('--global-scale', default=1.0, help='Depth scale multiplier')
@click.option('--sharp-cubemap-size', type=int, default=1536,
              help='Cubemap face size for SHARP (default 1536)')
@click.option('--sharp-projection', type=click.Choice(['cubemap', 'icosahedral']),
              default='icosahedral', help='Projection mode for SHARP refinement')
@click.option('--force-erp', is_flag=True, help='Process even if aspect ratio isn\'t 2:1')
@click.option('--batch', is_flag=True, help='Process all images in input directory')
@click.option('--device', default='cuda', help='Device: cuda, cpu, mps')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
@click.option('--mock-dap', is_flag=True, help='Use mock DAP model (for testing)')
@click.option('--refine', is_flag=True, default=False,
              help='Run GSFix3D disocclusion repair after conversion')
@click.option('--refine-iterations', default=3, type=int,
              help='Max refinement iterations (default: 3)')
@click.option('--refine-cameras', default=36, type=int,
              help='Novel-view cameras for hole detection (default: 36)')
@click.option('--generator', type=click.Choice(['da360', 'dap', 'sharp360']),
              default=None, help='Generator mode: da360, dap, or sharp360 (overrides --depth-model)')
@click.option('--side-count', type=int, default=6,
              help='Number of faces for SHARP 360 projection (default: 6)')
@click.option('--seedvr2-upscale', is_flag=True,
              help='Upscale faces with SeedVR2 before SHARP prediction')
def convert(
    input_path: str,
    output_path: str,
    depth_model: str,
    sharp_refine: bool,
    stride: int,
    depth_min: float,
    depth_max: float,
    sky_threshold: float,
    outlier_pruning: float,
    global_scale: float,
    sharp_cubemap_size: int,
    sharp_projection: str,
    force_erp: bool,
    batch: bool,
    device: str,
    quiet: bool,
    mock_dap: bool,
    refine: bool,
    refine_iterations: int,
    refine_cameras: int,
    generator: str,
    side_count: int,
    seedvr2_upscale: bool,
):
    """
    Convert equirectangular panorama to Gaussian splat PLY.

    INPUT_PATH: Input ERP image or directory
    OUTPUT_PATH: Output PLY file or directory

    Default mode is SPAG (fast, depth-driven). Add --sharp-refine for
    higher quality per-face SHARP refinement.
    """
    from .core import SPAG4D

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not quiet:
        if generator == 'sharp360':
            mode = f"SHARP 360 (sides={side_count}{', SeedVR2 upscale' if seedvr2_upscale else ''})"
        elif sharp_refine:
            mode = "SHARP refined"
        else:
            mode = f"SPAG (stride={stride})"
        depth_label = (generator or depth_model).upper()
        click.echo(f"Loading SPAG-4D [{depth_label} + {mode}]...")

    converter = SPAG4D(
        device=device,
        depth_model=depth_model,
        use_mock_dap=mock_dap,
        sharp_refine=sharp_refine,
        sharp_cubemap_size=sharp_cubemap_size,
        sharp_projection_mode=sharp_projection,
        generator=generator,
    )

    def run_single(img_path, out_path):
        return converter.convert(
            input_path=str(img_path),
            output_path=str(out_path),
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            stride=stride,
            outlier_pruning=outlier_pruning,
            global_scale=global_scale,
            force_erp=force_erp,
            generator=generator or depth_model,
            side_count=side_count,
            seedvr2_upscale=seedvr2_upscale,
        )

    if batch:
        if not input_path.is_dir():
            raise click.ClickException("Input path must be a directory for batch mode")

        output_path.mkdir(parents=True, exist_ok=True)

        image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_exts]

        if not quiet:
            click.echo(f"Processing {len(images)} images...")

        for img_path in images:
            out_path = output_path / (img_path.stem + '.ply')
            try:
                result = run_single(img_path, out_path)
                if not quiet:
                    click.echo(f"  {img_path.name} -> {result.splat_count:,} splats")
            except Exception as e:
                click.echo(f"  {img_path.name}: {e}", err=True)
    else:
        result = run_single(input_path, output_path)

        if not quiet:
            click.echo(f"Converted: {result.splat_count:,} Gaussians")
            click.echo(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
            click.echo(f"Time: {result.processing_time:.2f}s")
            click.echo(f"Depth range: {result.depth_range[0]:.2f}m - {result.depth_range[1]:.2f}m")

        if refine:
            if not quiet:
                click.echo("Running GSFix3D refinement...")

            from .refine import refine_splat
            import numpy as np

            depth_npy_path = str(output_path).replace('.ply', '_depth.npy')
            if not Path(depth_npy_path).exists():
                click.echo("Warning: depth .npy not found, skipping refinement", err=True)
            else:
                depth_map = np.load(depth_npy_path)
                refined_path = str(output_path).replace('.ply', '_refined.ply')

                refine_result = refine_splat(
                    ply_path=str(output_path),
                    panorama_path=str(input_path),
                    depth_map=depth_map,
                    max_iterations=refine_iterations,
                    num_cameras=refine_cameras,
                    output_path=refined_path,
                )

                if not quiet:
                    click.echo(f"Refined: holes {refine_result['initial_hole_fraction']:.2%}"
                               f" -> {refine_result['final_hole_fraction']:.2%}")
                    click.echo(f"Saved to: {refined_path}")


@main.command('download-models')
@click.option('--model', type=click.Choice(['dap', 'da360', 'gsfix3d', 'sharp', 'seedvr2', 'all']),
              default='all', help='Which model weights to download')
@click.option('--verify', is_flag=True, help='Verify downloaded weights')
def download_models(model: str, verify: bool):
    """Download and cache model weights."""
    if model in ('dap', 'all'):
        from .dap_model import DAPModel
        click.echo("Downloading DAP model weights...")
        try:
            path = DAPModel._get_or_download_weights()
            click.echo(f"DAP weights cached at: {path}")
            if verify:
                if DAPModel._verify_checksum(Path(path)):
                    click.echo("Checksum verified")
                else:
                    click.echo("Checksum verification skipped (no reference hash)")
        except Exception as e:
            click.echo(f"DAP download failed: {e}", err=True)
            if model == 'dap':
                raise click.Abort()

    if model in ('da360', 'all'):
        try:
            from .da360_model import DA360Model
            click.echo("Downloading DA360 model weights...")
            path = DA360Model._get_or_download_weights()
            click.echo(f"DA360 weights cached at: {path}")
        except ImportError:
            click.echo("DA360 model not yet available (architecture files needed)", err=True)
        except Exception as e:
            click.echo(f"DA360 download failed: {e}", err=True)
            if model == 'da360':
                raise click.Abort()

    if model in ('gsfix3d', 'all'):
        click.echo("Downloading GSFix3D checkpoint...")
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                "goldoak1421/gsfixer-full-replica-room1",
                local_dir="pretrained/gsfix3d",
            )
            click.echo(f"GSFix3D checkpoint cached at: {path}")
        except ImportError:
            click.echo("huggingface_hub not installed. Install with: pip install huggingface-hub", err=True)
        except Exception as e:
            click.echo(f"GSFix3D download failed: {e}", err=True)
            if model == 'gsfix3d':
                raise click.Abort()

    if model in ('sharp', 'all'):
        click.echo("SHARP model: auto-downloads on first use via Hugging Face Hub.")
        click.echo("No manual download required.")

    if model in ('seedvr2', 'all'):
        click.echo("SeedVR2 requires manual installation.")
        click.echo("Please follow the instructions at: https://github.com/TencentARC/SeedVR")
        click.echo("Install the package and place weights in pretrained/seedvr2/ before using --seedvr2-upscale.")


@main.command()
@click.option('--port', default=7860, help='Server port')
@click.option('--host', default='127.0.0.1', help='Server host')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(port: int, host: str, reload: bool):
    """Start the web UI server."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed. Install with: pip install uvicorn"
        )

    import logging
    import copy
    from uvicorn.config import LOGGING_CONFIG

    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find("GET /api/status") == -1

    log_config = copy.deepcopy(LOGGING_CONFIG)

    if 'filters' not in log_config:
        log_config['filters'] = {}

    log_config['filters']['endpoint_filter'] = {
        '()': EndpointFilter
    }

    if 'uvicorn.access' in log_config['loggers']:
        if 'filters' not in log_config['loggers']['uvicorn.access']:
            log_config['loggers']['uvicorn.access']['filters'] = []
        log_config['loggers']['uvicorn.access']['filters'].append("endpoint_filter")

    from api import kill_existing_server
    kill_existing_server(port)

    click.echo(f"Starting SPAG-4D web UI at http://{host}:{port}")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_config=log_config
    )


if __name__ == '__main__':
    main()
