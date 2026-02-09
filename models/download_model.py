# from modelscope import snapshot_download

# # Downloads Qwen3-0.6B to the specified directory
# snapshot_download(
#     model_id='Qwen/Qwen3-0.6B', 
#     local_dir='/home/tiago/huawei/models/Qwen3-0.6B'
# )


from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="/home/tiago/huawei/models/Qwen3-0.6B"
)