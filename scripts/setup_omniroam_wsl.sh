#!/usr/bin/env bash
# setup_omniroam_wsl.sh
# Run INSIDE WSL2: wsl bash scripts/setup_omniroam_wsl.sh
#
# Sets up OmniRoam and its dependencies in a conda environment.
# Safe to re-run (idempotent). Respects OMNIROAM_DIR env var override.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
OMNIROAM_DIR="${OMNIROAM_DIR:-$HOME/OmniRoam}"
CONDA_ENV="omniroam"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.1.0"
CUDA_VERSION="cu118"
OMNIROAM_REPO="https://github.com/yuhengliu02/OmniRoam.git"

# Model weight directories
WEIGHTS_ROOT="$OMNIROAM_DIR/weights"
OMNIROAM_WEIGHTS_DIR="$WEIGHTS_ROOT/OmniRoam_Preview"
WAN_WEIGHTS_DIR="$WEIGHTS_ROOT/Wan2.1-T2V-1.3B"

# ─── Helpers ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # no colour

ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
info() { echo -e "${CYAN}[--]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

section() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $*${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
}

# ─── Step 1: NVIDIA GPU check ─────────────────────────────────────────────────
section "Step 1/9 — NVIDIA GPU check"

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Ensure WSL2 NVIDIA drivers are installed on the Windows host. See: https://docs.nvidia.com/cuda/wsl-user-guide/"
fi

if ! nvidia-smi &>/dev/null; then
    fail "nvidia-smi failed. GPU may not be accessible from WSL2."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
ok "GPU detected: $GPU_NAME (driver $DRIVER_VERSION)"

# ─── Step 2: Install Miniconda ────────────────────────────────────────────────
section "Step 2/9 — Miniconda"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_ROOT/bin/conda"

if [ -f "$CONDA_BIN" ]; then
    ok "Miniconda already installed at $CONDA_ROOT"
else
    info "Downloading Miniconda installer..."
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -o "$MINICONDA_INSTALLER"
    info "Installing Miniconda to $CONDA_ROOT ..."
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_ROOT"
    rm -f "$MINICONDA_INSTALLER"
    ok "Miniconda installed"
fi

# Initialise conda for this shell session without modifying .bashrc permanently
# (user can run 'conda init bash' separately if they want shell integration)
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
ok "conda $(conda --version) available"

# ─── Step 3: Clone OmniRoam ───────────────────────────────────────────────────
section "Step 3/9 — OmniRoam repository"

if [ -d "$OMNIROAM_DIR/.git" ]; then
    ok "OmniRoam already cloned at $OMNIROAM_DIR"
    info "Pulling latest changes..."
    git -C "$OMNIROAM_DIR" pull --ff-only || warn "Could not fast-forward pull (local changes?). Skipping."
else
    info "Cloning OmniRoam into $OMNIROAM_DIR ..."
    git clone "$OMNIROAM_REPO" "$OMNIROAM_DIR"
    ok "OmniRoam cloned"
fi

# ─── Step 4: Create conda environment ────────────────────────────────────────
section "Step 4/9 — conda environment '$CONDA_ENV'"

if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    ok "conda env '$CONDA_ENV' already exists"
else
    info "Creating conda env '$CONDA_ENV' with Python $PYTHON_VERSION ..."
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VERSION"
    ok "conda env '$CONDA_ENV' created"
fi

# Activate env for remaining steps
conda activate "$CONDA_ENV"
ok "Activated conda env: $CONDA_ENV (Python $(python --version))"

# ─── Step 5: Install PyTorch with CUDA 11.8 ──────────────────────────────────
section "Step 5/9 — PyTorch $PYTORCH_VERSION + CUDA 11.8"

if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    ok "PyTorch $TORCH_VER already installed with CUDA support"
else
    info "Installing PyTorch $PYTORCH_VERSION with $CUDA_VERSION ..."
    pip install \
        torch=="${PYTORCH_VERSION}+${CUDA_VERSION}" \
        torchvision \
        torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

    if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
        ok "PyTorch installed with CUDA support"
    else
        fail "PyTorch installed but torch.cuda.is_available() is False. Check CUDA drivers."
    fi
fi

# Install requirements.txt from OmniRoam repo
REQ_FILE="$OMNIROAM_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    info "Installing OmniRoam requirements.txt ..."
    pip install -r "$REQ_FILE"
    ok "requirements.txt installed"
else
    warn "No requirements.txt found at $REQ_FILE — skipping"
fi

# ─── Step 6: Install Rust/Cargo ───────────────────────────────────────────────
section "Step 6/9 — Rust / Cargo (needed by DiffSynth-Studio)"

if command -v cargo &>/dev/null; then
    ok "Rust/Cargo already installed: $(cargo --version)"
else
    info "Installing Rust via rustup (non-interactive) ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
    # Source cargo env for this session
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
    ok "Rust installed: $(cargo --version)"
fi

# Ensure cargo is on PATH for the rest of this script
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

# ─── Step 7: Install DiffSynth-Studio ────────────────────────────────────────
section "Step 7/9 — DiffSynth-Studio"

if python -c "import diffsynth" &>/dev/null; then
    ok "DiffSynth-Studio already importable"
else
    info "Installing DiffSynth-Studio via pip ..."
    pip install git+https://github.com/modelscope/DiffSynth-Studio.git
    if python -c "import diffsynth" &>/dev/null; then
        ok "DiffSynth-Studio installed"
    else
        fail "DiffSynth-Studio install failed — 'import diffsynth' not working"
    fi
fi

# ─── Step 8: Model weight directories ────────────────────────────────────────
section "Step 8/9 — Model weight directories"

mkdir -p "$OMNIROAM_WEIGHTS_DIR"
mkdir -p "$WAN_WEIGHTS_DIR"
ok "Weight directories created:"
info "  OmniRoam Preview : $OMNIROAM_WEIGHTS_DIR"
info "  Wan2.1 T2V 1.3B  : $WAN_WEIGHTS_DIR"

# ── Download instructions ──────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Model weights — MANUAL DOWNLOAD REQUIRED                   ║${NC}"
echo -e "${YELLOW}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}║  1. OmniRoam Preview weights                                ║${NC}"
echo -e "${YELLOW}║     Source : https://huggingface.co/yuhengliu02/OmniRoam    ║${NC}"
echo -e "${YELLOW}║     Place in: $OMNIROAM_WEIGHTS_DIR${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}║     huggingface-cli download yuhengliu02/OmniRoam           ║${NC}"
echo -e "${YELLOW}║       --local-dir \"$OMNIROAM_WEIGHTS_DIR\"  ║${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}║  2. Wan2.1 T2V 1.3B weights                                 ║${NC}"
echo -e "${YELLOW}║     Source : https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B  ║${NC}"
echo -e "${YELLOW}║     Place in: $WAN_WEIGHTS_DIR${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}║     huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B         ║${NC}"
echo -e "${YELLOW}║       --local-dir \"$WAN_WEIGHTS_DIR\"  ║${NC}"
echo -e "${YELLOW}║                                                              ║${NC}"
echo -e "${YELLOW}║  Tip: pip install huggingface_hub[cli]                      ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"

# ─── Step 9: Verification ────────────────────────────────────────────────────
section "Step 9/9 — Verification"

PASS=0
FAIL=0

check() {
    local label="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        ok "$label"
        PASS=$((PASS + 1))
    else
        warn "FAIL — $label"
        FAIL=$((FAIL + 1))
    fi
}

check "Python 3.10"                    "python --version 2>&1 | grep -q '3\.10'"
check "torch importable"               "python -c 'import torch'"
check "torch.cuda.is_available()"      "python -c 'import torch; assert torch.cuda.is_available()'"
check "diffsynth importable"           "python -c 'import diffsynth'"
check "OmniRoam repo present"          "[ -d '$OMNIROAM_DIR/.git' ]"
check "OmniRoam weights dir exists"    "[ -d '$OMNIROAM_WEIGHTS_DIR' ]"
check "Wan2.1 weights dir exists"      "[ -d '$WAN_WEIGHTS_DIR' ]"
check "cargo available"                "command -v cargo"

# Print CUDA device info
echo ""
info "CUDA device info:"
python - <<'PYEOF'
import torch
print(f"  torch version   : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}           : {p.name} ({p.total_memory // 1024**2} MB)")
PYEOF

# ─── OmniRoamConfig values ───────────────────────────────────────────────────
echo ""
echo -e "${CYAN}─── OmniRoamConfig values (for reference) ──────────────────────${NC}"
python - <<PYEOF
import os

omniroam_dir   = os.path.expanduser("${OMNIROAM_DIR}")
weights_root   = os.path.join(omniroam_dir, "weights")
omniroam_wts   = os.path.join(weights_root, "OmniRoam_Preview")
wan_wts        = os.path.join(weights_root, "Wan2.1-T2V-1.3B")
conda_env      = "${CONDA_ENV}"

print(f"  omniroam_dir          = '{omniroam_dir}'")
print(f"  omniroam_weights_path = '{omniroam_wts}'")
print(f"  wan_weights_path      = '{wan_wts}'")
print(f"  conda_env             = '{conda_env}'")
PYEOF

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All $PASS checks passed. OmniRoam environment is ready.${NC}"
    echo ""
    echo -e "  Activate with: ${CYAN}conda activate ${CONDA_ENV}${NC}"
    echo -e "  OmniRoam dir : ${CYAN}${OMNIROAM_DIR}${NC}"
    echo ""
    echo -e "${YELLOW}Remember to download model weights (see box above) before running inference.${NC}"
else
    echo -e "${YELLOW}$PASS checks passed, $FAIL check(s) failed.${NC}"
    echo "Review the warnings above and re-run this script after fixing issues."
    exit 1
fi
