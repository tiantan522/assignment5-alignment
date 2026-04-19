"""
Math Baseline Evaluation Script

Evaluates Qwen 2.5 Math 1.5B zero-shot performance on GSM8K using the r1_zero prompt.

Usage:
    python -m cs336_alignment.math_baseline \
        --model_path /data/a5-alignment/Qwen2.5-Math-1.5B \
        --data_path data/gsm8k/test.jsonl \
        --output_path outputs/math_baseline_results.jsonl \
        --top_n 30
        
    uv run python -m cs336_alignment.math_baseline \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --data_path data/gsm8k/test.jsonl \
    --output_path outputs/math_baseline_results.jsonl \
    --top_n 30 \
    --tensor_parallel_size 4
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Callable, List

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_r1_zero_prompt_template(prompt_path: str) -> str:
    """Load the r1_zero prompt template from disk."""
    with open(prompt_path, "r") as f:
        return f.read()


def load_gsm8k_examples(data_path: str, top_n: int = None) -> List[dict]:
    """
    Load GSM8K examples from a JSONL file.

    Each example has:
      - "question": the math problem text
      - "answer": solution string ending with "#### <answer>"

    Returns a list of dicts with keys: "question", "answer", "ground_truth"
    where "ground_truth" is the extracted final numeric answer.
    """
    examples = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if top_n is not None and i >= top_n:
                break
            example = json.loads(line.strip())
            # Extract the ground-truth answer after "####"
            raw_answer = example["answer"]
            if "####" in raw_answer:
                ground_truth = raw_answer.split("####")[-1].strip()
            else:
                ground_truth = raw_answer.strip()
            example["ground_truth"] = ground_truth
            examples.append(example)
    return examples


def format_prompts(examples: List[dict], prompt_template: str) -> List[str]:
    """
    Format examples as string prompts using the r1_zero prompt template.

    The template contains `{question}` placeholder and ends with
    `<think>` so the model continues from there.
    """
    prompts = []
    for ex in examples:
        prompt = prompt_template.format(question=ex["question"])
        prompts.append(prompt)
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
    examples: List[dict] = None,
) -> dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    Args:
        vllm_model: A vLLM LLM instance.
        reward_fn: A callable that takes (response, ground_truth) and returns
                   a dict with keys: format_reward, answer_reward, reward.
        prompts: List of formatted prompt strings.
        ground_truths: List of ground truth answer strings (parallel to prompts).
        eval_sampling_params: vLLM SamplingParams for generation.
        output_path: Path to save results JSONL file.
        examples: Optional original examples (for serialization).

    Returns:
        A dict of aggregate evaluation metrics.
    """
    print(f"Generating outputs for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_reward = 0.0

    # Category counts
    correct_both = 0       # format=1, answer=1
    format_only = 0        # format=1, answer=0
    neither = 0            # format=0, answer=0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        # Compute rewards
        rewards = reward_fn(generated_text, ground_truth)
        format_reward = rewards["format_reward"]
        answer_reward = rewards["answer_reward"]
        reward = rewards["reward"]

        total_format_reward += format_reward
        total_answer_reward += answer_reward
        total_reward += reward

        if format_reward == 1.0 and answer_reward == 1.0:
            correct_both += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            format_only += 1
        else:
            neither += 1

        result = {
            "index": i,
            "question": examples[i]["question"] if examples else None,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "generation": generated_text,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "reward": reward,
        }
        results.append(result)

    n = len(prompts)
    metrics = {
        "n_examples": n,
        "correct_both_format_and_answer": correct_both,
        "format_reward_1_answer_reward_0": format_only,
        "format_reward_0_answer_reward_0": neither,
        "mean_format_reward": total_format_reward / n,
        "mean_answer_reward": total_answer_reward / n,
        "mean_reward": total_reward / n,
        "accuracy": correct_both / n,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples evaluated: {n}")
    print(f"  Correct (format=1, answer=1): {correct_both} ({correct_both/n*100:.1f}%)")
    print(f"  Format only (format=1, answer=0): {format_only} ({format_only/n*100:.1f}%)")
    print(f"  Neither (format=0, answer=0): {neither} ({neither/n*100:.1f}%)")
    print(f"Mean format reward: {metrics['mean_format_reward']:.4f}")
    print(f"Mean answer reward: {metrics['mean_answer_reward']:.4f}")
    print(f"Accuracy (answer correct): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print("=" * 60)

    # Serialize results to disk
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Save metrics separately
    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults serialized to: {output_path}")
    print(f"Metrics serialized to: {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen 2.5 Math 1.5B zero-shot on GSM8K using r1_zero prompt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/a5-alignment/Qwen2.5-Math-1.5B",
        help="Path to the Qwen 2.5 Math 1.5B model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to GSM8K test JSONL file",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="cs336_alignment/prompts/r1_zero.prompt",
        help="Path to the r1_zero prompt template",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/math_baseline_results.jsonl",
        help="Path to save evaluation results JSONL",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of examples to evaluate (default: 30)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism (default: 8 for P4d A100 instance)",
    )
    args = parser.parse_args()

    # Load prompt template
    print(f"Loading prompt template from: {args.prompt_path}")
    prompt_template = load_r1_zero_prompt_template(args.prompt_path)

    # Load data
    print(f"Loading data from: {args.data_path} (top {args.top_n} examples)")
    examples = load_gsm8k_examples(args.data_path, top_n=args.top_n)
    print(f"Loaded {len(examples)} examples")

    # Format prompts
    prompts = format_prompts(examples, prompt_template)
    ground_truths = [ex["ground_truth"] for ex in examples]

    # Print first prompt as a sanity check
    print("\n--- Example prompt (first example) ---")
    print(prompts[0])
    print("--- Ground truth:", ground_truths[0], "---\n")

    # Initialize vLLM model
    print(f"Loading model from: {args.model_path}")
    print(f"Using tensor_parallel_size={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # Set up sampling parameters
    # Per problem instructions:
    #   stop = ["</answer>"]
    #   include_stop_str_in_output = True
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Run evaluation
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=args.output_path,
        examples=examples,
    )

    # Release GPU memory so this script can be imported/called multiple times
    # in the same process (e.g., when reusing evaluate_vllm for later problems).
    print("\nReleasing GPU memory...")
    destroy_model_parallel()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory released.")

    return metrics


if __name__ == "__main__":
    main()
