import torch
from spag4d.spherical_grid import create_spherical_grid
from spag4d.gaussian_converter import equirect_to_gaussians

def test_scales():
    h, w = 200, 400
    img = torch.zeros(h, w, 3)
    depth = torch.full((h, w), 50.0)
    grid = create_spherical_grid(h, w, 'cpu', stride=1)
    
    # Run conversion without sky dome
    res = equirect_to_gaussians(img, depth, grid, sky_threshold=1000.0)
    
    # Check scales at row 0 (zenith) vs row 100 (equator)
    # The output is flattened, so we must calculate indices or just look at min/max of scales
    scales = res['scales']
    print("Scales min:", scales.min(dim=0)[0])
    print("Scales max:", scales.max(dim=0)[0])
    print("Scales mean:", scales.mean(dim=0))
    
    # Let's see the first few (near pole) vs middle (equator)
    # Note: poles might be merged if pole_rows > 0
    print("\nScale 0 (near pole):", scales[0])
    print("Scale mid (equator):", scales[len(scales)//2])

if __name__ == "__main__":
    test_scales()
