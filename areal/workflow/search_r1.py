"""Search-R1 workflow: multi-turn search-augmented RL with info masking.

Implements the same interaction loop as verl's LLMGenerationManager:
  1. Model generates text (possibly containing <search>query</search>)
  2. If search: call retrieval API, inject <information>results</information>
  3. If <answer>...</answer>: done, compute reward
  4. Repeat up to max_turns

Key equivalence with verl:
  - info_mask (zeros for <information> blocks) is represented via loss_mask
    (0 for observation tokens, 1 for model-generated tokens)
  - Postprocessing truncates at </search> or </answer> and re-encodes
  - max_start_length, max_prompt_length, max_obs_length caps are preserved
"""

import re
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("SearchR1Workflow")

INVALID_ACTION_MSG = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
    "Let me try again.\n"
)


class SearchR1Workflow(RolloutWorkflow):
    """Multi-turn search-augmented rollout workflow for Search-R1."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        max_turns: int = 2,
        max_start_length: int = 2048,
        max_prompt_length: int = 4096,
        max_response_length: int = 500,
        max_obs_length: int = 500,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)

        self.max_turns = max_turns
        self.max_start_length = max_start_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_obs_length = max_obs_length
        self.search_url = search_url
        self.topk = topk

        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    # -- Postprocessing helpers (mirror verl's generation.py) --

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, str]:
        """Extract action and content from <search> or <answer> tags."""
        pattern = r"<(search|answer)>(.*?)</\1>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, ""

    @staticmethod
    def _postprocess(output_str: str) -> tuple[str | None, str, str]:
        """Truncate output at </search> or </answer> and extract action.

        Returns (action, content, truncated_str).
        """
        if "</search>" in output_str:
            truncated = output_str.split("</search>")[0] + "</search>"
            action, content = SearchR1Workflow._parse_action(truncated)
            return action, content, truncated
        elif "</answer>" in output_str:
            truncated = output_str.split("</answer>")[0] + "</answer>"
            action, content = SearchR1Workflow._parse_action(truncated)
            return action, content, truncated
        return None, "", output_str

    def _postprocess_final(self, output_str: str) -> str:
        """Truncate final output (no search) at </answer>."""
        if "</answer>" in output_str:
            return output_str.split("</answer>")[0] + "</answer>"
        return output_str

    def _encode_truncated(
        self, truncated_str: str, original_tokens: list[int]
    ) -> list[int]:
        """Re-encode truncated string to token IDs.

        Matches verl's approach of decode -> truncate -> re-encode
        to ensure clean token boundaries at tag boundaries.
        """
        return list(
            self.tokenizer.encode(truncated_str, add_special_tokens=False)
        )

    # -- Search helpers (mirror verl's generation.py) --

    async def _search(self, query: str) -> str:
        """Call the search API and format results."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.search_url,
                json={"queries": [query], "topk": self.topk, "return_scores": True},
            )
        results = resp.json()["result"][0]
        return self._passages_to_string(results)

    @staticmethod
    def _passages_to_string(retrieval_result: list) -> str:
        """Format search results identically to verl's _passages2string."""
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    # -- Core episode logic --

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # Lazy-load reward function if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # 1. Tokenize prompt from data["messages"]
        messages = data["messages"]
        input_ids = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        input_ids = input_ids[-self.max_start_length :]  # cap

        # 2. Initialize tracking
        seq: list[int] = list(input_ids)
        logprobs: list[float] = [0.0] * len(input_ids)
        loss_mask: list[int] = [0] * len(input_ids)  # prompt: no loss
        versions: list[int] = [-1] * len(input_ids)
        prompt_len = len(input_ids)
        n_search_calls = 0
        n_invalid_actions = 0
        has_answer = False

        # 3. Multi-turn search loop
        for _turn in range(self.max_turns):
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(
                    n_samples=1, max_new_tokens=self.max_response_length
                ),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            # Postprocess: decode, truncate, extract action
            output_str = self.tokenizer.decode(
                resp.output_tokens, skip_special_tokens=True
            )
            action, content, truncated_str = self._postprocess(output_str)
            truncated_ids = self._encode_truncated(truncated_str, resp.output_tokens)

            if action == "search":
                n_search_calls += 1
                # Call search API and inject observation
                search_results = await self._search(content)
                obs_str = (
                    f"\n\n<information>{search_results}</information>\n\n"
                )
                obs_ids = list(
                    self.tokenizer.encode(obs_str, add_special_tokens=False)
                )
                if len(obs_ids) > self.max_obs_length:
                    obs_ids = obs_ids[: self.max_obs_length]

                # Append: model response (loss=1) + observation (loss=0)
                resp_len = len(truncated_ids)
                seq += truncated_ids + obs_ids
                logprobs += list(resp.output_logprobs[:resp_len]) + [0.0] * len(
                    obs_ids
                )
                loss_mask += [1] * resp_len + [0] * len(obs_ids)
                versions += list(resp.output_versions[:resp_len]) + [-1] * len(
                    obs_ids
                )

                # Update input_ids for next turn (rolling state)
                input_ids = (input_ids + truncated_ids + obs_ids)[
                    -self.max_prompt_length :
                ]

            elif action == "answer":
                has_answer = True
                # Done - append response (loss=1)
                resp_len = len(truncated_ids)
                seq += truncated_ids
                logprobs += list(resp.output_logprobs[:resp_len])
                loss_mask += [1] * resp_len
                versions += list(resp.output_versions[:resp_len])
                break

            else:
                n_invalid_actions += 1
                # Invalid action - append full response (loss=1) + error (loss=0)
                obs_ids = list(
                    self.tokenizer.encode(
                        INVALID_ACTION_MSG, add_special_tokens=False
                    )
                )
                seq += list(resp.output_tokens) + obs_ids
                logprobs += list(resp.output_logprobs) + [0.0] * len(obs_ids)
                loss_mask += [1] * resp.output_len + [0] * len(obs_ids)
                versions += list(resp.output_versions) + [-1] * len(obs_ids)
                input_ids = (input_ids + list(resp.output_tokens) + obs_ids)[
                    -self.max_prompt_length :
                ]
        else:
            # Loop exhausted without answer - final generation (no search)
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(
                    n_samples=1, max_new_tokens=self.max_response_length
                ),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)
            output_str = self.tokenizer.decode(
                resp.output_tokens, skip_special_tokens=True
            )
            truncated_str = self._postprocess_final(output_str)
            truncated_ids = self._encode_truncated(
                truncated_str, resp.output_tokens
            )
            resp_len = len(truncated_ids)
            seq += truncated_ids
            logprobs += list(resp.output_logprobs[:resp_len])
            loss_mask += [1] * resp_len
            versions += list(resp.output_versions[:resp_len])

        # 4. Compute reward
        completions_str = self.tokenizer.decode(seq[prompt_len:])
        prompt_str = self.tokenizer.decode(seq[:prompt_len])
        solution_str = prompt_str + completions_str
        ground_truth = {"target": data.get("golden_answers", [])}

        from areal.reward.qa_em import compute_score_em_with_metrics

        reward, reward_metrics = compute_score_em_with_metrics(
            solution_str=solution_str, ground_truth=ground_truth
        )

        # Compute length metrics
        total_len = len(seq)
        response_len = sum(loss_mask)  # tokens with loss_mask=1 (model output)
        obs_len = total_len - prompt_len - response_len  # observation tokens

        # Log all metrics
        tracker = stats_tracker.get("rollout")
        tracker.scalar(
            reward=reward,
            n_search_calls=n_search_calls,
            n_invalid_actions=n_invalid_actions,
            has_answer=float(has_answer),
            valid_format=reward_metrics["valid_format"],
            answer_correct=reward_metrics["answer_correct"],
            retrieval_correct=reward_metrics["retrieval_correct"],
            sequence_length=total_len,
            response_length=response_len,
            obs_length=obs_len,
        )

        # 5. Return tensor dict with batch dim 1
        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
