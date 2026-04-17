"""Format-aware exact match reward for Search-R1 QA tasks.

Ported from verl's qa_em_format.py with 4-tier scoring:
  - Correct answer + valid format  -> 1.0
  - Correct answer + invalid format -> 1 - format_weight (default 0.6)
  - Wrong/no answer + valid format  -> format_weight (default 0.4)
  - Wrong/no answer + invalid format -> 0.0

format_weight (lambda) defaults to 0.4 per the Search-R1 paper.
"""

import re
import string

# Tag names used in the Search-R1 interaction format
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_SEARCH_OPEN = "<search>"
_SEARCH_CLOSE = "</search>"
_INFO_OPEN = "<information>"
_INFO_CLOSE = "</information>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if prediction is None:
        return 0
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str):
    """Extract the answer from the solution string.

    Returns the content of the LAST <answer>...</answer> tag only if there
    are 2+ matches (the first one is typically in the prompt's example text).
    Returns None for 0 or 1 matches.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if len(matches) <= 1:
        return None

    return matches[-1].group(1).strip()


def is_valid_sequence(text):
    """Validate the think->search->information->...->answer sequence structure.

    Checks that:
    1. <|im_start|>assistant marker is present
    2. All think/search/information/answer tags are balanced
    3. Tags follow the expected sequence pattern
    4. No extraneous content between tags
    """
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    start_pos = assistant_match.end()
    content = text[start_pos:]

    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return (
                False,
                f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags",
            )

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # State machine: start -> in_think -> after_think -> in_search -> after_search
    #                 -> in_information -> information -> ... -> in_answer -> end
    # After information, we can go back to in_think (another search round)
    # or to in_answer (final answer)
    state = "start"

    for part in parts:
        if not part.strip():
            continue

        is_tag = re.match(r"</?(?:think|search|information|answer)>", part)

        if is_tag:
            if part == _THINK_OPEN and state in ("start", "information"):
                state = "in_think"
            elif part == _THINK_CLOSE and state == "in_think":
                state = "after_think"
            elif part == _SEARCH_OPEN and state == "after_think":
                state = "in_search"
            elif part == _SEARCH_CLOSE and state == "in_search":
                state = "after_search"
            elif part == _INFO_OPEN and state == "after_search":
                state = "in_information"
            elif part == _INFO_CLOSE and state == "in_information":
                state = "information"
            elif part == _ANSWER_OPEN and state == "after_think":
                state = "in_answer"
            elif part == _ANSWER_CLOSE and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # Content between tags
            if state in ("in_think", "in_search", "in_information", "in_answer"):
                pass  # content allowed inside tags
            elif state in ("start", "after_think", "after_search", "information"):
                if part.strip():
                    return False, f"Unexpected content between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_information_blocks(text):
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text, golden_answers):
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    format_weight=0.4,
    score=1.0,
):
    """4-tier scoring function for exact match with format awareness.

    Args:
        solution_str: the full solution text (prompt + completion)
        ground_truth: dict with 'target' key containing list of golden answers
        format_weight: lambda - reward for valid format without correct answer (default 0.4)
        score: reward for correct answer (default 1.0)

    Returns:
        float: reward value in {0, format_weight, 1-format_weight, score}
    """
    is_valid, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)

    if em_check(answer, ground_truth["target"]):
        if is_valid:
            return score  # correct answer + valid format
        else:
            return score - format_weight  # correct answer
    elif is_valid:
        return format_weight  # valid format
    else:
        return 0.0


def compute_score_em_with_metrics(
    solution_str,
    ground_truth,
    format_weight=0.4,
    score=1.0,
):
    """Same as compute_score_em but also returns diagnostic metrics dict."""
    is_valid, _ = is_valid_sequence(solution_str)
    answer = extract_solution(solution_str=solution_str)
    answer_correct = bool(em_check(answer, ground_truth["target"]))
    retrieval_correct = False
    if is_valid:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    has_answer = answer is not None

    if answer_correct:
        reward = score if is_valid else score - format_weight
    elif is_valid:
        reward = format_weight
    else:
        reward = 0.0

    metrics = {
        "valid_format": float(is_valid),
        "has_answer": float(has_answer),
        "answer_correct": float(answer_correct),
        "retrieval_correct": float(retrieval_correct),
    }
    return reward, metrics


def qa_em_reward_fn(
    prompt, completions, prompt_ids, completion_ids, golden_answers, **kwargs
):
    """AReaL-compatible reward function for QA exact match with format awareness.

    Args:
        prompt: decoded prompt string
        completions: decoded completion string
        prompt_ids: prompt token IDs
        completion_ids: completion token IDs
        golden_answers: list of acceptable answers (from dataset)

    Returns:
        float: reward value
    """
    solution_str = prompt + completions
    ground_truth = {"target": golden_answers}
    return compute_score_em(solution_str=solution_str, ground_truth=ground_truth)
