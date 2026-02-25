# spag4d/cli.py
"""
Command-line interface for SPAG-4D.
"""

import click
from pathlib import Path
from typing import Optional


@click.group()
@click.version_option(version="0.2.0")
def main():
    """SPAG-4D: Convert 360° panoramas to 3D Gaussian Splats."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--stride', default=2, help='Downsampling factor: 1, 2, 4, 8')
@click.option('--scale-factor', default=1.5, help='Gaussian scale multiplier')
@click.option('--thickness', default=0.1, help='Radial thickness ratio')
@click.option('--global-scale', default=1.0, help='Depth scale multiplier')
@click.option('--depth-min', default=0.1, help='Minimum depth in meters')
@click.option('--depth-max', default=100.0, help='Maximum depth in meters')
@click.option('--sky-threshold', default=80.0, help='Sky depth threshold (0 to disable)')
@click.option('--format', 'output_format', default='ply', 
              type=click.Choice(['ply', 'splat', 'both']),
              help='Output format')
@click.option('--sh-degree', default='0', type=click.Choice(['0', '3']),
              help='SH degree (0 or 3)')
@click.option('--force-erp', is_flag=True, help='Process even if aspect ratio isn\'t 2:1')
@click.option('--batch', is_flag=True, help='Process all images in input directory')
@click.option('--device', default='cuda', help='Device: cuda, cpu, mps')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
@click.option('--mock-dap', is_flag=True, help='Use mock DAP model (for testing)')
# Depth Model Options
@click.option('--depth-model', type=click.Choice(['panda', 'da3', 'dap']),
              default='panda', help='Depth model: panda (default), da3 (Depth Anything V3), or dap (legacy)')
@click.option('--guided-filter/--no-guided-filter', default=True,
              help='RGB-guided depth edge refinement (default: enabled)')
@click.option('--guided-strength', type=float, default=1.0,
              help='Strength of edge refinement (0.0-1.0)')
# SHARP Options
@click.option('--sharp-refine/--no-sharp-refine', default=True,
              help='Use SHARP model for perceptual attribute refinement (default: enabled)')
@click.option('--sharp-model', type=click.Path(exists=True), default=None,
              help='Path to SHARP checkpoint (downloads if not specified)')
@click.option('--sharp-cubemap-size', type=int, default=1536,
              help='Cubemap face size for SHARP (default 1536)')
@click.option('--sharp-projection', type=click.Choice(['cubemap', 'icosahedral']), 
              default='cubemap', help='SHARP projection mode: cubemap (6 faces) or icosahedral (20 faces)')
@click.option('--scale-blend', type=float, default=0.5,
              help='Blend factor for SHARP scales (0=geometric only, 1=SHARP only)')
@click.option('--opacity-blend', type=float, default=1.0,
              help='Blend factor for SHARP opacities')
@click.option('--sharp-depth-fuse/--no-sharp-depth-fuse', default=False,
              help='Use SHARP depth on cubemap faces for Phase 1 depth fusion')
def convert(
    input_path: str,
    output_path: str,
    stride: int,
    scale_factor: float,
    thickness: float,
    global_scale: float,
    depth_min: float,
    depth_max: float,
    sky_threshold: float,
    output_format: str,
    sh_degree: int,
    force_erp: bool,
    batch: bool,
    device: str,
    quiet: bool,
    mock_dap: bool,
    depth_model: str,
    guided_filter: bool,
    sharp_refine: bool,
    sharp_model: str,
    sharp_cubemap_size: int,
    sharp_projection: str,
    scale_blend: float,
    opacity_blend: float,
    guided_strength: float,
    sharp_depth_fuse: bool,
):
    """
    Convert equirectangular panorama to Gaussian splat.
    
    INPUT_PATH: Input ERP image or directory
    OUTPUT_PATH: Output PLY/SPLAT file or directory
    """
    from .core import SPAG4D
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Initialize converter
    if not quiet:
        click.echo("Loading SPAG-4D...")
    
    converter = SPAG4D(
        device=device,
        use_mock_dap=mock_dap,
        depth_model=depth_model,
        use_guided_filter=guided_filter,
        use_sharp_refinement=sharp_refine,
        sharp_model_path=sharp_model,
        sharp_cubemap_size=sharp_cubemap_size,
        sharp_projection_mode=sharp_projection  
    )
    
    if batch:
        # Batch mode: process all images in directory
        if not input_path.is_dir():
            raise click.ClickException("Input path must be a directory for batch mode")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_exts]
        
        if not quiet:
            click.echo(f"Processing {len(images)} images...")
        
        for img_path in images:
            out_name = img_path.stem + ('.splat' if output_format == 'splat' else '.ply')
            out_path = output_path / out_name
            
            try:
                result = converter.convert(
                    input_path=str(img_path),
                    output_path=str(out_path),
                    stride=stride,
                    scale_factor=scale_factor,
                    thickness_ratio=thickness,
                    global_scale=global_scale,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    sky_threshold=sky_threshold,
                    sh_degree=int(sh_degree),
                    output_format='splat' if output_format == 'splat' else 'ply',
                    force_erp=force_erp,
                    guided_strength=guided_strength,
                    scale_blend=scale_blend,
                    opacity_blend=opacity_blend,
                    sharp_depth_fuse=sharp_depth_fuse,
                )
                
                if not quiet:
                    click.echo(f"  ✓ {img_path.name} → {result.splat_count:,} splats")
                
                # Also generate SPLAT if format is 'both'
                if output_format == 'both':
                    splat_path = output_path / (img_path.stem + '.splat')
                    converter.convert_ply_to_splat(str(out_path), str(splat_path))
                    
            except Exception as e:
                click.echo(f"  ✗ {img_path.name}: {e}", err=True)
    
    else:
        # Single file mode
        fmt = 'splat' if output_format == 'splat' else 'ply'
        
        result = converter.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            stride=stride,
            scale_factor=scale_factor,
            thickness_ratio=thickness,
            global_scale=global_scale,
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            sh_degree=int(sh_degree),
            output_format=fmt,
            force_erp=force_erp,
            guided_strength=guided_strength,
            scale_blend=scale_blend,
            opacity_blend=opacity_blend,
            sharp_depth_fuse=sharp_depth_fuse,
        )
        
        if not quiet:
            click.echo(f"Converted: {result.splat_count:,} Gaussians")
            click.echo(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
            click.echo(f"Time: {result.processing_time:.2f}s")
            click.echo(f"Depth range: {result.depth_range[0]:.2f}m - {result.depth_range[1]:.2f}m")
        
        # Also generate SPLAT if format is 'both'
        if output_format == 'both':
            splat_path = Path(output_path).with_suffix('.splat')
            converter.convert_ply_to_splat(str(output_path), str(splat_path))
            if not quiet:
                click.echo(f"Also saved: {splat_path}")


@main.command('download-models')
@click.option('--verify', is_flag=True, help='Verify downloaded weights')
@click.option('--model', 'model_name', type=click.Choice(['panda', 'dap', 'all']),
              default='all', help='Which model to download (default: all)')
def download_models(verify: bool, model_name: str):
    """Download and cache depth model weights."""
    
    if model_name in ('panda', 'all'):
        from .panda_model import PanDAModel
        click.echo("Downloading PanDA model weights...")
        try:
            path = PanDAModel._get_or_download_weights()
            click.echo(f"✓ PanDA weights cached at: {path}")
        except Exception as e:
            click.echo(f"✗ PanDA download failed: {e}", err=True)
            if model_name == 'panda':
                raise click.Abort()
    
    if model_name in ('dap', 'all'):
        from .dap_model import DAPModel
        click.echo("Downloading DAP model weights...")
        try:
            path = DAPModel._get_or_download_weights()
            click.echo(f"✓ DAP weights cached at: {path}")
            if verify:
                if DAPModel._verify_checksum(Path(path)):
                    click.echo("✓ Checksum verified")
                else:
                    click.echo("⚠ Checksum verification skipped (no reference hash)")
        except Exception as e:
            click.echo(f"✗ DAP download failed: {e}", err=True)
            if model_name == 'dap':
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

    # Modify default log config
    log_config = copy.deepcopy(LOGGING_CONFIG)
    
    # Ensure filters list exists in access logger config (it might be None or missing)
    if 'filters' not in log_config:
        log_config['filters'] = {}
    
    log_config['filters']['endpoint_filter'] = {
        '()': EndpointFilter
    }
    
    if 'uvicorn.access' in log_config['loggers']:
        if 'filters' not in log_config['loggers']['uvicorn.access']:
             log_config['loggers']['uvicorn.access']['filters'] = []
        log_config['loggers']['uvicorn.access']['filters'].append("endpoint_filter")

    click.echo(f"Starting SPAG-4D web UI at http://{host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_config=log_config
    )


@main.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--fps', default=10, help='Frames per second to extract')
@click.option('--start', default=0.0, help='Start time in seconds')
@click.option('--duration', default=None, type=float, help='Duration in seconds')
@click.option('--stride', default=4, help='Downsampling factor')
@click.option('--stabilize', is_flag=True, help='Enable visual odometry stabilization')
@click.option('--device', default='cuda', help='Device')
@click.option('--sharp-refine/--no-sharp-refine', default=True,
              help='Use SHARP model for perceptual attribute refinement (default: enabled)')
@click.option('--sharp-cubemap-size', type=int, default=1536,
              help='Cubemap face size for SHARP (must be multiple of 384)')
@click.option('--sharp-projection', type=click.Choice(['cubemap', 'icosahedral']),
              default='cubemap', help='SHARP projection mode')
@click.option('--depth-model', type=click.Choice(['panda', 'da3', 'dap']),
              default='panda', help='Depth model: panda (default), da3 (Depth Anything V3), or dap (legacy)')
@click.option('--guided-filter/--no-guided-filter', default=True,
              help='RGB-guided depth edge refinement (default: enabled)')
@click.option('--scale-blend', type=float, default=0.5,
              help='Blend factor for SHARP scales (0=geometric only, 1=SHARP only)')
@click.option('--opacity-blend', type=float, default=1.0,
              help='Blend factor for SHARP opacities')
@click.option('--sharp-depth-fuse/--no-sharp-depth-fuse', default=False,
              help='Use SHARP depth on cubemap faces for Phase 1 depth fusion')
def convert_video(input_video: str, output_dir: str, fps: int, start: float, duration: float,
                  stride: int, stabilize: bool, device: str, sharp_refine: bool,
                  sharp_cubemap_size: int, sharp_projection: str,
                  depth_model: str, guided_filter: bool,
                  scale_blend: float, opacity_blend: float,
                  sharp_depth_fuse: bool):
    """
    Extract frames from 360° video and convert each to Gaussian splat.
    """
    import subprocess
    import tempfile
    import shutil
    from .core import SPAG4D
    
    input_video = Path(input_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visual odometry if stabilization enabled
    vo = None
    if stabilize:
        try:
            from spag4d.visual_odometry import SphericalVisualOdometry, transform_gaussians
            import cv2
            import numpy as np
            vo = SphericalVisualOdometry(n_features=1000)
            click.echo("Visual Odometry enabled")
        except ImportError:
            click.echo("⚠ scipy/opencv not installed, stabilization disabled", err=True)

    # Extract frames to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        click.echo(f"Extracting frames at {fps} FPS (start={start}s, dur={duration}s)...")
        
        cmd = [
            'ffmpeg',
            '-ss', str(start),
        ]
        if duration:
            cmd.extend(['-t', str(duration)])
            
        cmd.extend([
            '-i', str(input_video),
            '-vf', f'fps={fps}',
            '-qscale:v', '2',
            str(tmpdir / 'frame_%05d.jpg')
        ])
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        frames = sorted(tmpdir.glob('frame_*.jpg'))
        if not frames:
            click.echo("No frames extracted! Check start time/duration.", err=True)
            return
            
        click.echo(f"Extracted {len(frames)} frames")
        
        # Convert each frame
        converter = SPAG4D(
            device=device,
            depth_model=depth_model,
            use_guided_filter=guided_filter,
            use_sharp_refinement=sharp_refine,
            sharp_cubemap_size=sharp_cubemap_size,
            sharp_projection_mode=sharp_projection,
        )

        with click.progressbar(frames, label='Converting frames') as bar:
            for frame_path in bar:
                out_path = output_dir / (frame_path.stem + '.ply')
                try:
                    # Track camera motion for stabilization
                    if vo:
                        img_bgr = cv2.imread(str(frame_path))
                        vo.process_frame(img_bgr)

                    converter.convert(
                        input_path=str(frame_path),
                        output_path=str(out_path),
                        stride=stride,
                        force_erp=True,
                        scale_blend=scale_blend,
                        opacity_blend=opacity_blend,
                        sharp_depth_fuse=sharp_depth_fuse,
                    )

                    # Apply stabilization rotation to written PLY
                    if vo:
                        from spag4d.visual_odometry import transform_gaussians
                        from spag4d.ply_writer import load_ply_gaussians, save_ply_gaussians
                        R_stab = vo.get_stabilization_rotation()
                        gaussians = load_ply_gaussians(str(out_path))
                        gaussians = transform_gaussians(gaussians, R_stab)
                        save_ply_gaussians(gaussians, str(out_path))

                except Exception as e:
                    click.echo(f"\n⚠ {frame_path.name}: {e}", err=True)
    
    click.echo(f"DONE: Converted {len(frames)} frames to {output_dir}")


if __name__ == '__main__':
    main()
