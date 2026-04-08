#!/usr/bin/env bash
# scripts/regenerate_trajectory_snapshots.sh
# Regenerate snapshots from OmniRoam's actual function.
# Run: wsl bash scripts/regenerate_trajectory_snapshots.sh
set -euo pipefail
OMNIROAM_DIR="${OMNIROAM_DIR:-$HOME/OmniRoam}"
OUT_DIR="/mnt/d/SPAG-4D/tests/data/omniroam_trajectory_snapshots"
cd "$OMNIROAM_DIR"
conda run --no-banner -n omniroam python -c "
import sys, numpy as np
sys.path.insert(0, '.')
from infer_omniroam import make_cam_traj_from_preset_refspace
for preset in ['forward', 'backward', 'left', 'right', 's_curve', 'loop']:
    t = make_cam_traj_from_preset_refspace(preset=preset, step_m=0.25, amp_m=1.6, loop_radius_m=1.5)
    np.save(f'${OUT_DIR}/{preset}.npy', t.numpy())
    print(f'Saved {preset}.npy')
"
echo "Done."
