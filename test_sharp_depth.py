import torch
import numpy as np
import sys
import os

def log(msg):
    print(f"[TEST] {msg}")

def test_sharp_depth():
    try:
        from sharp.models import create_predictor, PredictorParams
    except ImportError:
        log("FAIL: ml-sharp not installed.")
        return False
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log("Initializing PredictorParams...")
    params = PredictorParams()
    
    log("Creating model...")
    model = create_predictor(params)
    
    # Load weights
    weight_path = "checkpoints/sharp.pt"
    # Or your download location for ML Sharp
    # Using SPAG-4D's sharp_refiner cache:
    from pathlib import Path
    cache_path = Path.home() / ".cache" / "spag4d" / "sharp" / "sharp.pt"
    if not cache_path.exists():
        log(f"FAIL: No SHARP weights found at {cache_path}")
        return False
        
    log("Loading weights...")
    state_dict = torch.load(cache_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    log("Running inference...")
    # Create dummy face [B, C, H, W]
    # Sharp's native size is 1536
    img = torch.rand(1, 3, 1536, 1536).to(device)
    f_px = 512.0
    disparity_factor = torch.tensor([f_px / 1536.0], device=device, dtype=torch.float32)
    
    with torch.inference_mode():
        # Step 1: Run the monodepth model directly
        # model is RGBGaussianPredictor
        log("Testing direct monodepth access...")
        monodepth_output = model.monodepth_model(img)
        disparity = monodepth_output.disparity
        
        log(f"Raw Disparity shape: {disparity.shape}")
        
        # Convert to depth
        depth = disparity_factor[:, None, None, None] / disparity.clamp(min=1e-4, max=1e4)
        log(f"Converted Depth shape: {depth.shape}")
        
        # Step 2: Test the DepthAlignment module
        # This handles local scaling
        aligned_depth, scale_map = model.depth_alignment(depth, None, monodepth_output.decoder_features)
        log(f"Aligned depth shape: {aligned_depth.shape}")
        
        # Get final output dimensions
        log(f"Max depth: {aligned_depth.max().item():.3f}, Min depth: {aligned_depth.min().item():.3f}")
        
    log("SUCCESS")
    return True

if __name__ == "__main__":
    if test_sharp_depth():
        sys.exit(0)
    else:
        sys.exit(1)
