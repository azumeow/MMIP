#!/usr/bin/env bash
# Run this INSIDE the V100 machine, from a transferred project directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU/driver unavailable.' >&2; exit 1; }
nvidia-smi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
if [ ! -x .venv/bin/python ]; then "$PYTHON_BIN" -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
# V100 supports fp16. CUDA 12.1 wheels require a compatible host driver.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1'
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check host driver and CUDA wheel"; print(torch.cuda.get_device_name(0))'
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
mkdir -p logs
if [ ! -f data/processed/train.jsonl ]; then python -u -m ml.prepare_data; fi
python -u -m ml.train_lr
# All examples in the training split, all encoder layers trainable; test untouched.
# Run through tmux/nohup as documented so a disconnected SSH session cannot kill training.
python -u -m ml.train_bert --device cuda --epochs 3 --batch-size 16 --max-length 192 --freeze-layers 0 --threads 4
python -m pytest -q
echo 'Training completed. Copy artifacts/ and reports/ back to your local project.'
