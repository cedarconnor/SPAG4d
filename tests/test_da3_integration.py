import sys
import os
import subprocess
import time
import torch
import shutil

def log(msg):
    print(f"[TEST] {msg}")

def test_da3():
    log("Testing Depth Anything V3 integration...")
    try:
        from spag4d.da3_model import DA3Model
        # Check if we can load it (this might trigger download)
        log("Import successful. Attempting to load model (metric)...")
        
        # Create dummy image [H, W, C]
        H, W = 512, 1024
        img = torch.rand(H, W, 3).cuda()
        
        # Load model
        model = DA3Model.load(variant="metric", device=torch.device("cuda"))
        log("Model loaded. Running inference...")
        
        # Run inference
        depth, mask = model.predict(img)
        log(f"Inference successful. Depth shape: {depth.shape}")
        
        if depth.shape != (H, W):
            log(f"FAIL: Output shape mismatch! Expected {(H, W)}, got {depth.shape}")
            return False
            
        log(f"Depth stats: min={depth.min():.2f}, max={depth.max():.2f}")
        return True
            
    except ImportError as e:
        log(f"FAIL: DA3 Not installed or Import Error: {e}")
        return False
    except Exception as e:
        log(f"FAIL: Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_da3_perspective_with_base_model():
    log("Testing Depth Anything V3 in perspective mode with DAP base anchor...")
    try:
        from spag4d.da3_model import DA3Model
        from spag4d.dap_model import MockDAPModel
        
        # Create dummy image [H, W, C]
        H, W = 512, 1024
        img = torch.rand(H, W, 3).cuda()
        
        # Load mock DAP base model
        base_dap = MockDAPModel.load(device=torch.device("cuda"))
        
        # Load DA3 model
        da3_model = DA3Model.load(variant="metric", device=torch.device("cuda"))
        
        # Run inference using perspective mode and base_model
        depth, mask = da3_model.predict(img, projection_mode="cubemap", base_model=base_dap)
        log(f"Inference successful with DAP global anchor. Depth shape: {depth.shape}")
        
        if depth.shape != (H, W):
            log(f"FAIL: Output shape mismatch! Expected {(H, W)}, got {depth.shape}")
            return False
            
        return True
            
    except ImportError as e:
        log(f"FAIL: DA3 Not installed or Import Error: {e}")
        return False
    except Exception as e:
        log(f"FAIL: Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bat_logic():
    log("Testing SPAG4D.bat logic (Server Startup)...")
    # The bat file basically activates venv and runs:
    # python -m spag4d.cli serve --port 7860
    
    # Use a random port to verify safe startup
    cmd = [sys.executable, "-m", "spag4d.cli", "serve", "--port", "7865"]
    
    log(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        # Wait 10 seconds to ensure it starts up and doesn't crash
        time.sleep(10)
        
        if proc.poll() is not None:
            # It exited
            out, err = proc.communicate()
            log(f"FAIL: Server process exited early via code {proc.returncode}")
            log(f"Stdout: {out}")
            log(f"Stderr: {err}")
            return False
        else:
            log("SUCCESS: Server is still running after 10s.")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return True
            
    except Exception as e:
        log(f"FAIL: Exception executing server: {e}")
        if proc.poll() is None:
            proc.kill()
        return False

def main():
    if not torch.cuda.is_available():
        log("WARNING: CUDA not available, strictly speaking this test requires CUDA for SPAG-4D models.")
        
    da3_ok = test_da3()
    da3_cube_ok = test_da3_perspective_with_base_model()
    bat_ok = test_bat_logic()
    
    if da3_ok and da3_cube_ok and bat_ok:
        log("ALL TESTS PASSED.")
        sys.exit(0)
    else:
        log("SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
