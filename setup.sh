#!/bin/bash
# setup.sh — run once on the pod after SSH access is available
# Usage: bash setup.sh <PANGRAM_API_KEY>
set -e

PANGRAM_API_KEY="${1:-}"

echo "=== humanizer-autoresearch pod setup ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Persist env vars
echo "export PANGRAM_API_KEY='${PANGRAM_API_KEY}'" >> ~/.bashrc
echo "export HF_HOME='/workspace/hf_cache'"        >> ~/.bashrc
echo "export TRANSFORMERS_CACHE='/workspace/hf_cache'" >> ~/.bashrc
echo "export HUGGINGFACE_HUB_CACHE='/workspace/hf_cache'" >> ~/.bashrc
source ~/.bashrc

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Flash attention 2 (significant throughput boost on H100)
pip install flash-attn --no-build-isolation -q

# Project directory
cd /workspace
git clone https://github.com/txmed82/humanizer-autoresearch.git || true
cd /workspace/humanizer-autoresearch

# Install deps
uv sync

# Pre-download Llama 3.1 8B (saves time on first experiment)
echo "Pre-downloading Llama-3.1-8B-Instruct weights..."
uv run python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct',
                                     cache_dir='/workspace/hf_cache')
print('Tokenizer OK')
# Download weights only (don't load into VRAM yet)
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct',
                  cache_dir='/workspace/hf_cache',
                  ignore_patterns=['*.safetensors.index.json'])
print('Model weights cached.')
"

# Prepare data
echo "Preparing training and eval data..."
uv run prepare.py

echo ""
echo "=== Setup complete ==="
echo "Run a baseline experiment:"
echo "  uv run train.py > run.log 2>&1 && grep '^humanize_score:' run.log"
