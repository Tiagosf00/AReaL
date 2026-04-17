"""Tests for Search-R1 reward function, dataset, and workflow components."""

import os
import sys

import pytest

from areal.reward.qa_em import (
    compute_score_em,
    compute_score_em_with_metrics,
    em_check,
    extract_information_blocks,
    extract_solution,
    is_retrieval_correct,
    is_valid_sequence,
    normalize_answer,
    qa_em_reward_fn,
)
from areal.workflow.search_r1 import SearchR1Workflow

TO = '<think>'
TC = '</think>'


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("The Quick Brown Fox") == "quick brown fox"
        assert normalize_answer("An Apple A Day") == "apple day"


class TestEmCheck:
    def test_exact_match(self):
        assert em_check("Paris", ["Paris"]) == 1

    def test_case_insensitive(self):
        assert em_check("paris", ["Paris"]) == 1

    def test_no_match(self):
        assert em_check("London", ["Paris"]) == 0

    def test_multiple_golden(self):
        assert em_check("Paris", ["Paris", "paris france"]) == 1

    def test_string_golden(self):
        assert em_check("42", "42") == 1


class TestExtractSolution:
    def test_no_answer_tag(self):
        assert extract_solution("some text without answer") is None

    def test_one_answer_tag(self):
        assert extract_solution("text <answer>42</answer>") is None

    def test_two_answer_tags(self):
        result = extract_solution("<answer>first</answer> text <answer>second</answer>")
        assert result == "second"

    def test_three_answer_tags(self):
        result = extract_solution(
            "<answer>a</answer> <answer>b</answer> <answer>c</answer>"
        )
        assert result == "c"

    def test_multiline_answer(self):
        result = extract_solution(
            "<answer>first</answer>\n<answer>multi\nline</answer>"
        )
        assert result == "multi\nline"


class TestIsValidSequence:
    def test_valid_simple(self):
        text = f"<|im_start|>assistant\n{TO}reasoning{TC}\n<answer>42</answer>"
        is_valid, _ = is_valid_sequence(text)
        assert is_valid is True

    def test_valid_with_search(self):
        text = (
            f"<|im_start|>assistant\n"
            f"{TO}reasoning{TC}\n"
            f"<search>query</search>\n"
            f"<information>result</information>\n"
            f"{TO}more{TC}\n"
            f"<answer>42</answer>"
        )
        is_valid, _ = is_valid_sequence(text)
        assert is_valid is True

    def test_missing_assistant_marker(self):
        text = f"{TO}reasoning{TC}\n<answer>42</answer>"
        is_valid, _ = is_valid_sequence(text)
        assert is_valid is False

    def test_unbalanced_tags(self):
        text = f"<|im_start|>assistant\n{TO}reasoning\n<answer>42</answer>"
        is_valid, _ = is_valid_sequence(text)
        assert is_valid is False

    def test_wrong_order(self):
        text = "<|im_start|>assistant\n<answer>42</answer>"
        is_valid, _ = is_valid_sequence(text)
        assert is_valid is False


class TestExtractInformationBlocks:
    def test_single_block(self):
        text = "before <information>content</information> after"
        blocks = extract_information_blocks(text)
        assert blocks == ["content"]

    def test_multiple_blocks(self):
        text = "<information>first</information> mid <information>second</information>"
        blocks = extract_information_blocks(text)
        assert blocks == ["first", "second"]


class TestIsRetrievalCorrect:
    def test_correct_retrieval(self):
        text = "<information>The answer is Paris, France</information>"
        assert is_retrieval_correct(text, ["Paris"]) is True

    def test_incorrect_retrieval(self):
        text = "<information>London is the capital of UK</information>"
        assert is_retrieval_correct(text, ["Paris"]) is False


class TestComputeScoreEM:
    def test_no_answer_invalid_format(self):
        assert compute_score_em("no answer here", {"target": ["42"]}) == 0.0

    def test_correct_answer_valid_format(self):
        prompt = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        model_output = f"{TO}reasoning{TC}\n<answer>42</answer>"
        full_trajectory = prompt + model_output
        assert extract_solution(full_trajectory) == "42"
        assert is_valid_sequence(full_trajectory)[0] is True
        assert compute_score_em(full_trajectory, {"target": ["42"]}) == 1.0

    def test_wrong_answer_valid_format(self):
        prompt = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        model_output = f"{TO}reasoning{TC}\n<answer>wrong</answer>"
        full_trajectory = prompt + model_output
        assert compute_score_em(full_trajectory, {"target": ["42"]}) == 0.4

    def test_correct_answer_invalid_format(self):
        text = "some text <answer>first</answer> <answer>42</answer>"
        assert compute_score_em(text, {"target": ["42"]}) == 0.6

    def test_no_answer_valid_format(self):
        text = f"<|im_start|>assistant\n{TO}reasoning{TC}\n<answer>42</answer>"
        assert compute_score_em(text, {"target": ["42"]}) == 0.4

    def test_custom_format_weight(self):
        prompt = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        model_output = f"{TO}reasoning{TC}\n<answer>wrong</answer>"
        full_trajectory = prompt + model_output
        assert compute_score_em(full_trajectory, {"target": ["42"]}, format_weight=0.3) == 0.3


class TestQaEmRewardFn:
    def test_basic_reward(self):
        prompt = (
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        completions = f"{TO}thinking{TC}\n<answer>42</answer>"
        reward = qa_em_reward_fn(
            prompt, completions, [], [], golden_answers=["42"]
        )
        assert reward == 1.0


class TestParseAction:
    def test_search_action(self):
        action, content = SearchR1Workflow._parse_action(
            "Let me search <search>Frank Spring NHL</search>"
        )
        assert action == "search"
        assert content == "Frank Spring NHL"

    def test_answer_action(self):
        action, content = SearchR1Workflow._parse_action(
            "The answer is <answer>Cleveland Barons</answer>"
        )
        assert action == "answer"
        assert content == "Cleveland Barons"

    def test_invalid_action(self):
        action, content = SearchR1Workflow._parse_action("just some text")
        assert action is None
        assert content == ""


class TestPostprocess:
    def test_truncate_at_search_close(self):
        output = "thinking <search>query</search> extra text"
        action, content, truncated = SearchR1Workflow._postprocess(output)
        assert action == "search"
        assert content == "query"
        assert "extra text" not in truncated
        assert truncated.endswith("</search>")

    def test_truncate_at_answer_close(self):
        output = "reasoning <answer>42</answer> more text"
        action, content, truncated = SearchR1Workflow._postprocess(output)
        assert action == "answer"
        assert content == "42"
        assert "more text" not in truncated
        assert truncated.endswith("</answer>")

    def test_no_tags(self):
        output = "just plain text without any tags"
        action, content, truncated = SearchR1Workflow._postprocess(output)
        assert action is None
        assert content == ""
        assert truncated == output

    def test_search_before_answer(self):
        output = "<search>q</search> info <answer>42</answer>"
        action, content, truncated = SearchR1Workflow._postprocess(output)
        assert action == "search"
        assert content == "q"


class TestPassagesToString:
    def test_format_single_result(self):
        result = [
            {"document": {"id": "1", "contents": "Title\nLine1\nLine2"}, "score": 10.0}
        ]
        output = SearchR1Workflow._passages_to_string(result)
        assert output == "Doc 1(Title: Title) Line1\nLine2\n"

    def test_format_multiple_results(self):
        result = [
            {"document": {"id": "1", "contents": "T1\nL1"}, "score": 10.0},
            {"document": {"id": "2", "contents": "T2\nL2"}, "score": 9.0},
        ]
        output = SearchR1Workflow._passages_to_string(result)
        assert "Doc 1(Title: T1) L1" in output
        assert "Doc 2(Title: T2) L2" in output


class TestDatasetLoading:
    @pytest.mark.skipif(
        not os.path.exists(
            "/home/d00940872_/XDS_2025/data/nq_search/train_original.parquet"
        ),
        reason="NQ search dataset not available",
    )
    def test_load_train_dataset(self):
        from areal.dataset.nq_search import get_nq_search_rl_dataset

        ds = get_nq_search_rl_dataset(
            path="/home/d00940872_/XDS_2025/data/nq_search",
            split="train",
        )
        assert len(ds) > 0
        item = ds[0]
        assert "messages" in item
        assert "golden_answers" in item
        assert "data_source" in item
        assert isinstance(item["messages"], list)
        assert isinstance(item["golden_answers"], list)


class TestVerlEquivalence:
    @pytest.mark.skipif(
        not os.path.exists(
            "/home/d00940872_/Search-R1/verl/utils/reward_score/qa_em_format.py"
        ),
        reason="Search-R1 verl not available",
    )
    def test_reward_equivalence(self):
        sys.path.insert(0, "/home/d00940872_/Search-R1")
        try:
            from verl.utils.reward_score.qa_em_format import (
                compute_score_em as verl_compute_score_em,
            )
        except ImportError:
            pytest.skip("verl not importable")

        format_weight = 0.4
        test_cases = [
            ("no answer here", {"target": ["42"]}),
            (
                "text <answer>first</answer> more <answer>42</answer>",
                {"target": ["42"]},
            ),
        ]

        for solution_str, ground_truth in test_cases:
            areal_score = compute_score_em(
                solution_str, ground_truth, format_weight=format_weight
            )
            verl_score = verl_compute_score_em(
                solution_str,
                ground_truth,
                structure_format_score=format_weight,
                final_format_score=0,
                retrieval_score=0,
                format_score=0,
                score=1.0,
            )
            assert areal_score == verl_score, (
                f"Mismatch: areal={areal_score} vs verl={verl_score}"
            )


class TestComputeScoreEMWithMetrics:
    def test_correct_answer_valid_format(self):
        prompt = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        model_output = f"{TO}reasoning{TC}\n<answer>42</answer>"
        full_trajectory = prompt + model_output
        reward, metrics = compute_score_em_with_metrics(
            full_trajectory, {"target": ["42"]}
        )
        assert reward == 1.0
        assert metrics["valid_format"] == 1.0
        assert metrics["has_answer"] == 1.0
        assert metrics["answer_correct"] == 1.0

    def test_wrong_answer_valid_format(self):
        prompt = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        model_output = f"{TO}reasoning{TC}\n<answer>wrong</answer>"
        full_trajectory = prompt + model_output
        reward, metrics = compute_score_em_with_metrics(
            full_trajectory, {"target": ["42"]}
        )
        assert reward == 0.4
        assert metrics["valid_format"] == 1.0
        assert metrics["answer_correct"] == 0.0

    def test_no_answer_invalid_format(self):
        reward, metrics = compute_score_em_with_metrics(
            "no answer here", {"target": ["42"]}
        )
        assert reward == 0.0
        assert metrics["valid_format"] == 0.0
        assert metrics["has_answer"] == 0.0
        assert metrics["answer_correct"] == 0.0

    def test_consistent_with_compute_score_em(self):
        """Verify reward matches compute_score_em for all tiers."""
        prompt = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\nQuestion. For example, <answer> Beijing </answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        cases = [
            (f"{TO}r{TC}\n<answer>42</answer>", {"target": ["42"]}),  # correct + valid
            (f"{TO}r{TC}\n<answer>w</answer>", {"target": ["42"]}),  # wrong + valid
            ("no answer", {"target": ["42"]}),  # no answer
        ]
        for model_output, gt in cases:
            full = prompt + model_output if model_output != "no answer" else model_output
            r1 = compute_score_em(full, gt)
            r2, _ = compute_score_em_with_metrics(full, gt)
            assert r1 == r2
