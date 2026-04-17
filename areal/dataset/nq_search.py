"""NQ Search dataset loader for Search-R1 RL training.

Loads parquet files from the nq_search dataset directory. The parquet files
are expected to be named {split}_original.parquet (e.g., train_original.parquet,
test_original.parquet).
"""

import os

import pandas as pd
from datasets import Dataset


def get_nq_search_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    **kwargs,
) -> Dataset:
    """Load nq_search parquet for RL training.

    Args:
        path: Directory containing the parquet files.
        split: "train" or "test".
        tokenizer: Tokenizer (unused, kept for API compatibility).
        max_length: Optional max sequence length filter.
        **kwargs: Additional keyword arguments.

    Returns:
        HuggingFace Dataset with items containing:
            - messages: List[dict] chat format for the prompt
            - golden_answers: List[str] acceptable answers
            - data_source: str identifying the data source
    """
    parquet_file = os.path.join(path, f"{split}_original.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"NQ search parquet not found: {parquet_file}")

    df = pd.read_parquet(parquet_file)
    data = []
    for _, row in df.iterrows():
        item = {
            "messages": list(row["prompt"]),
            "golden_answers": list(row["golden_answers"]),
            "data_source": row.get("data_source", "nq"),
        }
        data.append(item)

    return Dataset.from_list(data)
