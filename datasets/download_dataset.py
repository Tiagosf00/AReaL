from datasets import load_dataset
custom_path = "./datasets/gsm8k"
dataset = load_dataset("gsm8k", "main", cache_dir=custom_path)

print(f"Dataset downloaded to: {custom_path}")