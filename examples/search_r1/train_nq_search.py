"""Search-R1 GRPO training on NQ Search dataset with Qwen2.5-7B-Base.

Usage:
    python -m areal.experimental.trainer.rl --config-path examples/search_r1 --config-name nq_search_grpo_npu
"""

import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        reward_fn="areal.reward.qa_em.qa_em_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        max_turns=2,
        max_start_length=2048,
        max_prompt_length=4096,
        max_response_length=500,
        max_obs_length=500,
        search_url="http://127.0.0.1:8007/retrieve",
        topk=3,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.search_r1.SearchR1Workflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.search_r1.SearchR1Workflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
