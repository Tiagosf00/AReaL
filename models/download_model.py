from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./models/Qwen3-0.6B"
)