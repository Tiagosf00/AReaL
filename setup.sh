#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🚀 Starting environment setup..."

apt update
apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev curl git

curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.12.3
pyenv global 3.12.3

# 1. Install python libraries
pip install huggingface_hub datasets pytest tensorboard

# 2. Install uv 
echo "📦 Installing uv..."
python -m pip install -U uv
uv sync --extra cuda

uv pip install -e . --no-deps

# uv run python3 areal/tools/validate_installation.py


# 3. Run download scripts
echo "📥 Downloading datasets..."
python3 datasets/download_dataset.py

echo "📥 Downloading models..."
python3 models/download_model.py

echo "✅ Setup complete!"

# 4. Run tests

uv run python -m areal.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo_lora.yaml

uv run python -m areal.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml