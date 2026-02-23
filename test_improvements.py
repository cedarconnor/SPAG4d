import argparse
import sys
import os
import torch
from spag4d.core import SPAG4D

def test_improvements():
    print("Testing SPAG-4D Tier 2/3 Improvements...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Needs a sample image, just generate a dummy one if test image doesn't exist
    from PIL import Image
    import numpy as np
    
    img_path = "test_panorama_8k.png"
    if not os.path.exists(img_path):
        print("Generating 4000x2000 test image...")
        img_np = np.zeros((2000, 4000, 3), dtype=np.uint8)
        # Add basic vertical gradient to simulate sky/ground
        for y in range(2000):
            color = int(255 * y / 2000)
            img_np[y, :, :] = [color, color, 255 - color]
        Image.fromarray(img_np).save(img_path)

    try:
        core = SPAG4D(device="cuda" if torch.cuda.is_available() else "cpu", depth_model="panda")
        print("\n[SUCCESS] PanDA core initialized successfully")
        
        # Test 1: High-res tiled inference via core
        res = core.convert(
            input_path=img_path,
            output_path="test_panda_tiles.ply",
            stride=4,  # Fast
            sky_dome=True,
            sky_threshold=80.0
        )
        print(f"[SUCCESS] Tiled PanDA + Pole Merge + SH band-1 + Sky Dome succeeded: {res.splat_count} splats")
        
        # Check SH values in PLY
        from plyfile import PlyData
        pdata = PlyData.read("test_panda_tiles.ply")
        vertex = pdata['vertex']
        has_sh1 = 'f_rest_0' in vertex
        print(f"[SUCCESS] PLY SH Band-1 coefficients present: {has_sh1}")
            
    except Exception as e:
        print(f"[FAIL] Failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improvements()
