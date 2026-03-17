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
              default='dap', help='Depth estimation model')
@click.option('--sharp-refine', is_flag=True,
              help='Enable SHARP per-face refinement (slower, higher quality)')
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
        mode = "SHARP refined" if sharp_refine else f"SPAG (stride={stride})"
        click.echo(f"Loading SPAG-4D [{depth_model.upper()} + {mode}]...")

    converter = SPAG4D(
        device=device,
        depth_model=depth_model,
        use_mock_dap=mock_dap,
        sharp_refine=sharp_refine,
        sharp_cubemap_size=sharp_cubemap_size,
        sharp_projection_mode=sharp_projection,
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


@main.command('download-models')
@click.option('--model', type=click.Choice(['dap', 'da360', 'all']),
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
