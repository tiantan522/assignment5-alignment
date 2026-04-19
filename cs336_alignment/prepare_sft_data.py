"""
Prepare SFT training data from HuggingFace dataset.

Downloads:
  eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1

Filters and transforms:
  1. Swaps <response> tags to <answer> tags in the `cot` field.
  2. Keeps only examples where `cot` contains both <think>...</think>
     and <answer>...</answer> (correct format).
  3. Keeps only examples where the answer extracted from `cot` matches
     the ground-truth in the `answer` field.

Output:
  A .jsonl file with fields:
    {"prompt": <original question>, "response": <filtered/transformed cot>}

Usage:
  python -m cs336_alignment.prepare_sft_data \
      --output_path data/sft_train.jsonl \
      --split train
"""

import argparse
import json
import os
import re

from datasets import load_dataset

# Import the grader to reuse the answer-grading logic
from cs336_alignment.drgrpo_grader import grade, extract_answer


# ── Format check ──────────────────────────────────────────────────────────────

def has_correct_format(cot: str) -> bool:
    """Return True if cot has the correct r1_zero format:
        <think> ... </think> <answer> ... </answer>

    Checks:
      - Starts with (or contains) an opening <think> tag
      - Contains the exact '</think> <answer>' boundary that r1_zero_reward_fn uses
      - Contains a closing </answer> tag

    Note: '</think> <answer>' already implies </think> is present; we additionally
    require <think> to ensure the thinking block was actually opened.
    """
    return (
        "<think>" in cot
        and "</think> <answer>" in cot
        and "</answer>" in cot
    )


def swap_response_to_answer(cot: str) -> str:
    """Replace <response> / </response> tags with <answer> / </answer>.

    Also normalizes the boundary between </think> and <answer> to use a
    single space (as required by r1_zero_reward_fn), handling cases where
    the dataset uses a newline or other whitespace between the two tags.
    """
    cot = cot.replace("<response>", "<answer>")
    cot = cot.replace("</response>", "</answer>")
    # Normalize "</think>\n<answer>" (and any whitespace variant) →
    # "</think> <answer>" to match r1_zero_reward_fn's exact format check.
    cot = re.sub(r"</think>\s+<answer>", "</think> <answer>", cot)
    return cot


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer_from_cot(cot: str) -> str | None:
    """Extract the answer text between <answer> and </answer>.

    Handles these common formats in the dataset:
      - Plain number:      <answer>72</answer>
      - GSM8K hash prefix: <answer>#### 72</answer>
      - LaTeX boxed:       <answer>\\boxed{72}</answer>
    """
    match = re.search(r"<answer>(.*?)</answer>", cot, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        # Strip leading '####' (GSM8K-style annotation in the dataset)
        answer_text = re.sub(r"^#+\s*", "", answer_text).strip()
        # Further extract if it's wrapped in \boxed{}
        if "\\boxed" in answer_text:
            answer_text = extract_answer(answer_text)
        return answer_text if answer_text else None
    return None


def clean_answer_tag_content(cot: str) -> str:
    """Remove leading '####' and extra whitespace inside <answer>...</answer> tags.

    Ensures the model learns to produce clean numeric answers like:
      <answer>72</answer>
    instead of:
      <answer>#### 72</answer>
    """
    def _clean_match(m: re.Match) -> str:
        content = m.group(1).strip()
        # Strip leading hash marks and whitespace (e.g. "#### 72" → "72")
        content = re.sub(r"^#+\s*", "", content).strip()
        return f"<answer>{content}</answer>"

    return re.sub(r"<answer>(.*?)</answer>", _clean_match, cot, flags=re.DOTALL)


def extract_ground_truth(raw_answer: str) -> str:
    """Extract the numeric ground-truth after '####' in the GSM8K answer field."""
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip()
    return raw_answer.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def prepare_sft_data(
    output_path: str,
    split: str = "train",
    hf_dataset: str = "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1",
) -> None:
    print(f"Loading dataset: {hf_dataset} (split={split})")
    dataset = load_dataset(hf_dataset, split=split)
    print(f"Total examples in split: {len(dataset)}")

    n_total = len(dataset)
    n_format_ok = 0
    n_answer_correct = 0
    n_skipped_format = 0
    n_skipped_answer = 0

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as out_f:
        for example in dataset:
            question = example["question"]
            raw_answer = example["answer"]       # GSM8K-style: "... #### <number>"
            cot = example["cot"]                 # Chain-of-thought response

            # Step 1: Swap <response> → <answer> and normalize whitespace
            cot = swap_response_to_answer(cot)

            # Step 1b: Clean up answer tag content (strip #### prefix) so that
            # both the grader and the saved training data are free of hash annotations.
            cot = clean_answer_tag_content(cot)

            # Step 2: Keep only correctly formatted examples
            if not has_correct_format(cot):
                n_skipped_format += 1
                continue
            n_format_ok += 1

            # Step 3: Keep only examples where the cot answer matches ground truth
            ground_truth = extract_ground_truth(raw_answer)
            model_answer = extract_answer_from_cot(cot)

            if model_answer is None:
                n_skipped_answer += 1
                continue

            is_correct = grade(model_answer, ground_truth, fast=True)
            if not is_correct:
                n_skipped_answer += 1
                continue

            n_answer_correct += 1

            # Write the filtered example
            record = {
                "prompt": question,
                "response": cot,
            }
            out_f.write(json.dumps(record) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total examples:            {n_total}")
    print(f"After format filter:       {n_format_ok}  "
          f"(skipped {n_skipped_format} with bad format)")
    print(f"After answer correctness:  {n_answer_correct}  "
          f"(skipped {n_skipped_answer} with wrong answer)")
    print(f"Final kept:                {n_answer_correct}")
    print(f"Output written to:         {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare SFT training data from HuggingFace"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/sft_train.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1",
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    prepare_sft_data(
        output_path=args.output_path,
        split=args.split,
        hf_dataset=args.hf_dataset,
    )


if __name__ == "__main__":
    main()
