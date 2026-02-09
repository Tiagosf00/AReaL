# from modelscope import dataset_snapshot_download

# # Downloads the dataset to the specified directory
# dataset_snapshot_download(
#     dataset_id='modelscope/gsm8k',  # Replace with your target dataset ID
#     local_dir='/home/tiago/huawei/datasets/gsm8k'
# )




from datasets import load_dataset
custom_path = "./gsm8k"
dataset = load_dataset("gsm8k", "main", cache_dir=custom_path)

print(f"Dataset downloaded to: {custom_path}")