"""
SFT Training Script for Qwen 2.5 Math 1.5B.

Implements supervised fine-tuning (Algorithm 1) with:
  - Periodic vLLM evaluation on GSM8K validation set
  - WandB logging with separate train/eval x-axes
  - Gradient clipping (clip norm = 1.0)
  - Gradient accumulation
  - Support for varying dataset sizes: {128, 256, 512, 1024, full}

Usage:

  # Ablation: 128 examples
  python -m cs336_alignment.train_sft \
      --model_path Qwen/Qwen2.5-Math-1.5B \
      --train_data data/sft_train.jsonl \
      --val_data data/gsm8k/test.jsonl \
      --output_dir outputs/sft_128 \
      --n_train_examples 128 \
      --wandb_project cs336-sft
    
  # Full dataset SFT
  uv run python -m cs336_alignment.train_sft \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --train_data data/sft_train.jsonl \
    --val_data data/gsm8k/test.jsonl \
    --output_dir outputs/sft_full \
    --n_train_examples -1 \
    --train_gpu 0 --vllm_gpu 1 \
    --wandb_project cs336-sft
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from unittest.mock import patch

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import (
    evaluate_vllm,
    format_prompts,
    load_gsm8k_examples,
    load_r1_zero_prompt_template,
)
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """Initialize a vLLM instance on a specific GPU device."""
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL to allow single-GPU vLLM alongside training GPU
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Copy current policy weights into a vLLM instance for evaluation."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def release_vllm(llm: LLM) -> None:
    """Release vLLM GPU memory."""
    print("Releasing vLLM GPU memory...")
    destroy_model_parallel()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("vLLM GPU memory released.")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sft_data(
    data_path: str,
    n_examples: int = -1,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load SFT training data from a JSONL file.
    Each line: {"prompt": str, "response": str}
    """
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)

    if n_examples > 0:
        examples = examples[:n_examples]

    return examples


# ── Training ──────────────────────────────────────────────────────────────────

def run_sft(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_device = f"cuda:{args.train_gpu}"
    vllm_device = f"cuda:{args.vllm_gpu}"

    # ── WandB setup ───────────────────────────────────────────────────────────
    run_name = f"sft_n{args.n_train_examples}_lr{args.lr}_bs{args.batch_size}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # ── Load training data ────────────────────────────────────────────────────
    print(f"Loading training data from: {args.train_data}")
    train_examples = load_sft_data(
        args.train_data,
        n_examples=args.n_train_examples,
        shuffle=True,
        seed=args.seed,
    )
    print(f"Training on {len(train_examples)} examples")
    wandb.config.update({"actual_n_train": len(train_examples)}, allow_val_change=True)

    # ── Load validation data ──────────────────────────────────────────────────
    # load_gsm8k_examples adds "ground_truth" by extracting from "#### <num>"
    prompt_template = load_r1_zero_prompt_template(args.prompt_path)
    print(f"Loading validation data from: {args.val_data}")
    val_examples = load_gsm8k_examples(args.val_data, top_n=args.n_val_examples)
    val_prompts = format_prompts(val_examples, prompt_template)
    val_ground_truths = [ex["ground_truth"] for ex in val_examples]
    print(f"Validating on {len(val_prompts)} examples")

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load directly onto the training GPU to avoid the Flash Attention
    # "model not initialized on GPU" warning
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=train_device,
    )
    policy.train()

    # ── Optimizer & LR scheduler ──────────────────────────────────────────────
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_optimizer_steps = args.n_epochs * max(1, len(train_examples) // args.batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_optimizer_steps, eta_min=args.lr * 0.1)

    # ── vLLM for periodic evaluation ──────────────────────────────────────────
    print(f"Initializing vLLM on device: {vllm_device}")
    llm = init_vllm(
        model_id=args.model_path,
        device=vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    eval_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    gradient_accumulation_steps = max(1, args.batch_size // args.microbatch_size)
    train_step = 0
    eval_step = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.n_epochs):
        random.shuffle(train_examples)
        print(f"\n=== Epoch {epoch + 1}/{args.n_epochs} ===")

        for batch_start in range(0, len(train_examples), args.batch_size):
            batch = train_examples[batch_start : batch_start + args.batch_size]
            if not batch:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_n_response_tokens = 0

            # ── Microbatch gradient accumulation ──────────────────────────────
            for mb_start in range(0, len(batch), args.microbatch_size):
                mb = batch[mb_start : mb_start + args.microbatch_size]
                if not mb:
                    continue

                prompt_strs = [ex["prompt"] for ex in mb]
                response_strs = [ex["response"] for ex in mb]

                tokenized = tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
                input_ids = tokenized["input_ids"].to(train_device)
                labels = tokenized["labels"].to(train_device)
                response_mask = tokenized["response_mask"].to(train_device)

                log_probs_dict = get_response_log_probs(policy, input_ids, labels)
                policy_log_probs = log_probs_dict["log_probs"]

                loss, meta = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    normalize_constant=args.normalize_constant,
                )
                batch_loss += loss.item()
                batch_n_response_tokens += meta.get("n_response_tokens", 0)

            # ── Gradient clipping + optimizer step ────────────────────────────
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_step += 1

            wandb.log({
                "train/loss": batch_loss,
                "train/grad_norm": grad_norm.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/n_response_tokens": batch_n_response_tokens,
                "train_step": train_step,
            })

            if train_step % 10 == 0:
                print(f"  step {train_step:4d} | loss {batch_loss:.4f} | "
                      f"grad_norm {grad_norm:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

            # ── Periodic evaluation ───────────────────────────────────────────
            if train_step % args.eval_every == 0 or train_step == 1:
                print(f"\n  [Eval] step={train_step} — loading policy into vLLM...")
                policy.eval()
                with torch.no_grad():
                    load_policy_into_vllm_instance(policy, llm)
                policy.train()

                eval_output_path = os.path.join(
                    args.output_dir, f"eval_step{eval_step:05d}.jsonl"
                )
                metrics = evaluate_vllm(
                    vllm_model=llm,
                    reward_fn=r1_zero_reward_fn,
                    prompts=val_prompts,
                    ground_truths=val_ground_truths,
                    eval_sampling_params=eval_sampling_params,
                    output_path=eval_output_path,
                    examples=val_examples,
                )
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/mean_format_reward": metrics["mean_format_reward"],
                    "eval/mean_answer_reward": metrics["mean_answer_reward"],
                    "eval/n_correct": metrics["correct_both_format_and_answer"],
                    "eval_step": eval_step,
                })
                eval_step += 1
                print(f"  [Eval] accuracy={metrics['accuracy']:.4f}\n")

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}")
        print(f"Saving checkpoint to: {ckpt_dir}")
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n[Final Eval]")
    policy.eval()
    with torch.no_grad():
        load_policy_into_vllm_instance(policy, llm)
    final_eval_path = os.path.join(args.output_dir, "eval_final.jsonl")
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        eval_sampling_params=eval_sampling_params,
        output_path=final_eval_path,
        examples=val_examples,
    )
    wandb.log({
        "eval/accuracy": metrics["accuracy"],
        "eval/mean_format_reward": metrics["mean_format_reward"],
        "eval/mean_answer_reward": metrics["mean_answer_reward"],
        "eval_step": eval_step,
    })
    print(f"Final accuracy: {metrics['accuracy']:.4f}")

    # ── Release vLLM GPU memory ────────────────────────────────────────────────
    release_vllm(llm)
    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for Qwen 2.5 Math 1.5B")

    # Paths
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_path", type=str,
                        default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="outputs/sft")

    # Data
    parser.add_argument("--n_train_examples", type=int, default=-1,
                        help="Number of training examples (-1 = full dataset)")
    parser.add_argument("--n_val_examples", type=int, default=30,
                        help="Number of validation examples")

    # Training hyperparams
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Effective batch size (examples per optimizer step)")
    parser.add_argument("--microbatch_size", type=int, default=8,
                        help="Microbatch size for gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--normalize_constant", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Evaluate every N optimizer steps")

    # Sampling params (consistent with math_baseline.py)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate")

    # GPU assignment (2-GPU setup: train on GPU 0, vLLM on GPU 1)
    parser.add_argument("--train_gpu", type=int, default=0,
                        help="GPU index for policy training")
    parser.add_argument("--vllm_gpu", type=int, default=1,
                        help="GPU index for vLLM evaluation")
    # Lower utilization so the KV cache doesn't consume all GPU memory —
    # the policy weights (~3 GB) need to be loaded into vLLM at eval time.
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.50)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="cs336-sft")

    args = parser.parse_args()
    run_sft(args)


if __name__ == "__main__":
    main()
