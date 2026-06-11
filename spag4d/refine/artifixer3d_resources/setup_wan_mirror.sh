#!/usr/bin/env bash
# One-time: build the local Wan2.1 mirror at /data/wan_te so the ArtiFixer3D backend
# runs fully offline (HF_HUB_OFFLINE=1, --model_id /data/wan_te) and never touches the
# crashing HF xet native downloader (silent exit-255 in the cuda12 container).
#
# Vendored from experiments/artifixer_eval/_dl_wan_te.sh + _dl_vae.sh (Phase 1, byte-verified
# against the HF tree API). Transformer weights come from artifixer-14b.pt, so only the
# transformer config/index is staged here (NOT the 57GB of transformer shards).
#
# Run inside WSL:  bash setup_wan_mirror.sh
set -u
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

BASE="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/main"
DST=/data/wan_te
mkdir -p "$DST/text_encoder" "$DST/tokenizer" "$DST/vae" "$DST/transformer" "$DST/scheduler"

WGET="wget -c --tries=0 --retry-connrefused --waitretry=5 --read-timeout=120 --timeout=60 --progress=dot:giga"

# ── UMT5-XXL text encoder (5 shards) + tokenizer (for the caption embedding) ──
TE_FILES=(
  "config.json"
  "model.safetensors.index.json"
  "model-00001-of-00005.safetensors"
  "model-00002-of-00005.safetensors"
  "model-00003-of-00005.safetensors"
  "model-00004-of-00005.safetensors"
  "model-00005-of-00005.safetensors"
)
TOK_FILES=(
  "special_tokens_map.json"
  "spiece.model"
  "tokenizer.json"
  "tokenizer_config.json"
)

for f in "${TE_FILES[@]}"; do
  echo "### TE $f"
  $WGET -O "$DST/text_encoder/$f" "$BASE/text_encoder/$f" || { echo "FAIL TE $f"; exit 1; }
done
for f in "${TOK_FILES[@]}"; do
  echo "### TOK $f"
  $WGET -O "$DST/tokenizer/$f" "$BASE/tokenizer/$f" || { echo "FAIL TOK $f"; exit 1; }
done

# ── VAE (config + weights) + transformer/scheduler configs (weights from the .pt) ──
$WGET -O "$DST/vae/config.json"                      "$BASE/vae/config.json"
$WGET -O "$DST/vae/diffusion_pytorch_model.safetensors" "$BASE/vae/diffusion_pytorch_model.safetensors"
$WGET -O "$DST/transformer/config.json"              "$BASE/transformer/config.json"
$WGET -O "$DST/transformer/diffusion_pytorch_model.safetensors.index.json" "$BASE/transformer/diffusion_pytorch_model.safetensors.index.json"
$WGET -O "$DST/scheduler/scheduler_config.json"      "$BASE/scheduler/scheduler_config.json"

echo "=== /data/wan_te mirror ==="
find "$DST" -maxdepth 2 -type f -printf '%s\t%p\n' | sort -k2
echo "WAN_MIRROR_COMPLETE"
