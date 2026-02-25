import sys
import numpy as np
import torch
import traceback

def log(msg):
    print(f"[TEST] {msg}")

def test_sharp_depth_fusion():
    log("Testing SharpDepthFusion initialization...")
    try:
        from spag4d.sharp_depth_fusion import SharpDepthFusion
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test with lower resolution for speed, native is 1536 but it should handle resize properly
        face_size = 512
        fusion = SharpDepthFusion(device, face_size=face_size)
    except Exception as e:
        log(f"FAIL: Initialization error: {e}")
        traceback.print_exc()
        return False

    log("Testing SharpDepthFusion load_model...")
    try:
        fusion.load_model()
    except Exception as e:
        log(f"FAIL: Model load error: {e}")
        traceback.print_exc()
        return False

    log("Testing SharpDepthFusion fuse...")
    try:
        # Create dummy ERP image and depth
        H, W = 512, 1024
        # Random uint8 image
        erp_image = np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
        # Random depth in [1, 10] meters
        dap_depth = np.random.uniform(1.0, 10.0, size=(H, W)).astype(np.float32)

        fused_depth, confidence = fusion.fuse(erp_image, dap_depth)
        
        log(f"Fused Depth Shape: {fused_depth.shape}")
        log(f"Confidence Shape: {confidence.shape}")
        
        if fused_depth.shape != (H, W):
            log(f"FAIL: Expected fused_depth shape {(H, W)}, got {fused_depth.shape}")
            return False
            
        if confidence.shape != (H, W):
            log(f"FAIL: Expected confidence shape {(H, W)}, got {confidence.shape}")
            return False
            
        log(f"Fused Depth Min: {fused_depth.min():.3f}, Max: {fused_depth.max():.3f}")
        log(f"Confidence Min: {confidence.min():.3f}, Max: {confidence.max():.3f}")
            
    except Exception as e:
        log(f"FAIL: Fusion error: {e}")
        traceback.print_exc()
        return False
        
    log("SUCCESS: SharpDepthFusion is fully functional.")
    return True

if __name__ == "__main__":
    if test_sharp_depth_fusion():
        sys.exit(0)
    else:
        sys.exit(1)
