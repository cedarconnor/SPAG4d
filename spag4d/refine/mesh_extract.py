"""Mesh extraction from depth map for GSFixer dual conditioning."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_conditioning_mesh(depth_map, panorama, simplify_ratio=0.1):
    """Extract a rough textured mesh from equirectangular depth.

    Uses Open3D Poisson surface reconstruction, then simplifies.
    The mesh is a conditioning signal — needs approximate structure, not perfect geometry.
    """
    import open3d as o3d
    import trimesh

    h, w = depth_map.shape

    # SPAG convention (spherical_grid.py): θ decreases left-to-right,
    # φ is colatitude 0 (north pole) to π (south pole).
    # rhat = [sin(φ)*cos(θ), cos(φ), -sin(φ)*sin(θ)]
    theta = (1.0 - np.linspace(0, 1, w, endpoint=False)) * 2 * np.pi
    phi = np.linspace(0, np.pi, h)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = depth_map * np.sin(phi_grid) * np.cos(theta_grid)
    y = depth_map * np.cos(phi_grid)
    z = -depth_map * np.sin(phi_grid) * np.sin(theta_grid)

    stride = max(1, min(h, w) // 64)
    xs = x[::stride, ::stride].flatten()
    ys = y[::stride, ::stride].flatten()
    zs = z[::stride, ::stride].flatten()
    colors = panorama[::stride, ::stride, :3].reshape(-1, 3)

    valid = (depth_map[::stride, ::stride].flatten() > 0.01) & \
            (depth_map[::stride, ::stride].flatten() < 500)
    points = np.stack([xs[valid], ys[valid], zs[valid]], axis=1)
    colors = colors[valid]

    if len(points) < 100:
        logger.warning("Too few valid depth points for mesh extraction")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.clip(0, 1))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Simplify using Open3D (no extra dependencies) before converting to trimesh
    current_faces = len(mesh_o3d.triangles)
    target_faces = max(100, int(current_faces * simplify_ratio))
    if current_faces > target_faces:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_faces)

    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vertex_colors = np.asarray(mesh_o3d.vertex_colors) if mesh_o3d.has_vertex_colors() else None

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(vertex_colors * 255).astype(np.uint8) if vertex_colors is not None else None,
    )

    logger.info(f"Extracted mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def render_mesh(mesh, camera, resolution=(512, 512)):
    """Render mesh from a camera pose for dual conditioning.

    Falls back to gray placeholder if rendering fails.
    """
    if mesh is None:
        return np.ones((*resolution, 3), dtype=np.float32) * 0.5

    try:
        import trimesh
        import io
        from PIL import Image

        scene = trimesh.Scene(mesh)
        data = scene.save_image(resolution=resolution)
        img = np.array(Image.open(io.BytesIO(data))).astype(np.float32) / 255.0
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    except Exception as e:
        logger.warning(f"Mesh rendering failed ({e}), returning gray placeholder")
        return np.ones((*resolution, 3), dtype=np.float32) * 0.5
