#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🚀 Starting environment setup..."

# 1. Install python libraries
pip install uv huggingface_hub datasets pytest torch
pip install -e .

# 2. Install uv 
echo "📦 Installing uv..."
pip install uv
uv sync --extra cuda

# 3. Run download scripts
echo "📥 Downloading datasets..."
python3 datasets/download_dataset.py

echo "📥 Downloading models..."
python3 models/download_model.py

echo "✅ Setup complete!"


# Container image:
# ghcr.io/inclusionai/areal-runtime:v0.5.3