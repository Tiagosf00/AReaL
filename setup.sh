#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🚀 Setting up .venv environment..."

# 1. Create .venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 2. Activate the environment
source .venv/bin/activate

echo "🚀 Starting environment setup..."

# 3. Install python libraries (uv, snapshot_download, load_dataset)
pip install --user uv huggingface_hub datasets

# 4. Install uv 
echo "📦 Installing uv..."
pip install uv
uv sync --extra cuda

# 5. Run download scripts
echo "📥 Downloading datasets..."
python3 datasets/download_dataset.py

echo "📥 Downloading models..."
python3 models/download_model.py

echo "✅ Setup complete!"