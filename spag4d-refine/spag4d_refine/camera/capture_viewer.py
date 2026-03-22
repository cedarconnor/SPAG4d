"""Interactive camera keyframe capture using viser (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .pinhole import CameraSet, PinholeCamera


def launch_capture_viewer(
    ply_path: str | Path,
    vfov_deg: float = 60.0,
    width: int = 1920,
    height: int = 1080,
    port: int = 8890,
    output_path: Optional[str | Path] = None,
) -> CameraSet:
    """
    Launch an interactive viser viewer for camera keyframe capture.

    Requires the [viewer] optional dependency.

    Args:
        ply_path: Path to the Gaussian splat PLY file.
        vfov_deg: Vertical FOV for captured cameras.
        width: Camera width.
        height: Camera height.
        port: viser server port.
        output_path: If provided, save cameras.json on exit.

    Returns:
        CameraSet with captured keyframes.
    """
    try:
        import viser
    except ImportError:
        raise ImportError(
            "viser is required for interactive capture. "
            "Install with: pip install spag4d-refine[viewer]"
        )

    server = viser.ViserServer(port=port)
    camera_set = CameraSet()

    print(f"[Capture Viewer] Open http://localhost:{port} in your browser")
    print("[Capture Viewer] Press 'k' to capture a keyframe, 'q' to finish")

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle) -> None:
        capture_btn = client.gui.add_button("Capture Keyframe")
        done_btn = client.gui.add_button("Done")
        count_text = client.gui.add_text("Keyframes", initial_value="0")

        @capture_btn.on_click
        def _(_) -> None:
            # Get current camera transform from viser
            T = client.camera.wxyz  # quaternion
            pos = client.camera.position

            # Build c2w from viser's camera state
            import scipy.spatial.transform as st
            rot = st.Rotation.from_quat([T[1], T[2], T[3], T[0]])  # viser WXYZ → scipy XYZW
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rot.as_matrix()
            c2w[:3, 3] = pos

            cam = PinholeCamera.from_fov(vfov_deg, width, height, c2w)
            camera_set.append(cam)
            count_text.value = str(len(camera_set))
            print(f"  Captured keyframe {len(camera_set)} at {pos}")

        @done_btn.on_click
        def _(_) -> None:
            if output_path and len(camera_set) > 0:
                camera_set.save_json(output_path)
                print(f"  Saved {len(camera_set)} cameras to {output_path}")
            server.request_share_url()  # Signal to stop

    try:
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    if output_path and len(camera_set) > 0:
        camera_set.save_json(output_path)
        print(f"Saved {len(camera_set)} cameras to {output_path}")

    return camera_set
