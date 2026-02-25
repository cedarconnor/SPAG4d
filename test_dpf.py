import sys
import numpy as np
import torch
import traceback

def log(msg):
    print(f"[TEST] {msg}")

def test_depth_pro_fusion():
    try:
        from spag4d.depth_pro_fusion import DepthProFusion
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        log("Testing DepthProFusion initialization...")
        fusion = DepthProFusion(device, face_size=512)
        
        log("Testing DepthProFusion load_model...")
        fusion.load_model()
        
        log("Model loaded. Testing fuse...")
        H, W = 512, 1024
        erp_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        dap_depth = np.random.rand(H, W).astype(np.float32) + 0.1
        
        fused_depth, confidence = fusion.fuse(erp_img, dap_depth)
        
        if fused_depth.shape != (H, W):
            log(f"FAIL: Expected fused_depth shape {(H, W)}, got {fused_depth.shape}")
            return False
            
        if confidence.shape != (H, W):
            log(f"FAIL: Expected confidence shape {(H, W)}, got {confidence.shape}")
            return False
            
        log("SUCCESS: Depth Pro Fusion worked!")
        return True
    
    except ImportError as e:
        log(f"SKIP/FAIL: Could not import ml-depth-pro: {e}")
        return False
    except Exception as e:
        log(f"FAIL: Runtime Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_depth_pro_fusion():
        sys.exit(0)
    else:
        sys.exit(1)
