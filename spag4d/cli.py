# spag4d/cli.py
"""
Command-line interface for SPAG-4D.
"""

import click
from pathlib import Path


@click.group()
@click.version_option(version="2.0.0")
def main():
    """SPAG-4D: Convert 360° panoramas to 3D Gaussian Splats."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--depth-min', default=0.1, help='Minimum depth in meters')
@click.option('--depth-max', default=100.0, help='Maximum depth in meters')
@click.option('--sky-threshold', default=80.0, help='Sky depth threshold (0 to disable)')
@click.option('--grid-jitter', default=0.03, help='Grid jitter for anti-aliasing (0=off, 0.5=max)')
@click.option('--outlier-pruning', default=0.0, help='Outlier removal strength (0=off, 1=aggressive)')
@click.option('--global-scale', default=1.0, help='Depth scale multiplier')
@click.option('--sharp-cubemap-size', type=int, default=1536,
              help='Cubemap face size for SHARP (default 1536)')
@click.option('--sharp-projection', type=click.Choice(['cubemap', 'icosahedral']),
              default='cubemap', help='Projection mode: cubemap (6 faces) or icosahedral (20 faces)')
@click.option('--force-erp', is_flag=True, help='Process even if aspect ratio isn\'t 2:1')
@click.option('--batch', is_flag=True, help='Process all images in input directory')
@click.option('--device', default='cuda', help='Device: cuda, cpu, mps')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
@click.option('--mock-dap', is_flag=True, help='Use mock DAP model (for testing)')
def convert(
    input_path: str,
    output_path: str,
    depth_min: float,
    depth_max: float,
    sky_threshold: float,
    grid_jitter: float,
    outlier_pruning: float,
    global_scale: float,
    sharp_cubemap_size: int,
    sharp_projection: str,
    force_erp: bool,
    batch: bool,
    device: str,
    quiet: bool,
    mock_dap: bool,
):
    """
    Convert equirectangular panorama to Gaussian splat PLY.

    INPUT_PATH: Input ERP image or directory
    OUTPUT_PATH: Output PLY file or directory
    """
    from .core import SPAG4D

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not quiet:
        click.echo("Loading SPAG-4D...")

    converter = SPAG4D(
        device=device,
        use_mock_dap=mock_dap,
        sharp_cubemap_size=sharp_cubemap_size,
        sharp_projection_mode=sharp_projection,
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
                result = converter.convert(
                    input_path=str(img_path),
                    output_path=str(out_path),
                    depth_min=depth_min,
                    depth_max=depth_max,
                    sky_threshold=sky_threshold,
                    grid_jitter=grid_jitter,
                    outlier_pruning=outlier_pruning,
                    global_scale=global_scale,
                    force_erp=force_erp,
                )
                if not quiet:
                    click.echo(f"  {img_path.name} -> {result.splat_count:,} splats")
            except Exception as e:
                click.echo(f"  {img_path.name}: {e}", err=True)
    else:
        result = converter.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            grid_jitter=grid_jitter,
            outlier_pruning=outlier_pruning,
            global_scale=global_scale,
            force_erp=force_erp,
        )

        if not quiet:
            click.echo(f"Converted: {result.splat_count:,} Gaussians")
            click.echo(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
            click.echo(f"Time: {result.processing_time:.2f}s")
            click.echo(f"Depth range: {result.depth_range[0]:.2f}m - {result.depth_range[1]:.2f}m")


@main.command('download-models')
@click.option('--verify', is_flag=True, help='Verify downloaded weights')
def download_models(verify: bool):
    """Download and cache DAP depth model weights."""
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
        raise click.Abort()


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
