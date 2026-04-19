# CS336 Spring 2025 Assignment 5: Alignment

> **Goal:** Fine-tune [Qwen 2.5 Math 1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) for math reasoning on GSM8K using Supervised Fine-Tuning (SFT) and reinforcement learning (GRPO with verified rewards).

For the full assignment description, see:
- [cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)
- [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf) *(optional supplement on safety, instruction tuning, and RLHF)*

---

## Table of Contents

- [What's Been Implemented](#whats-been-implemented)
  - [1. SFT Data Preparation](#1-sft-data-preparation)
  - [2. SFT Core Utilities](#2-sft-core-utilities)
  - [3. SFT Training Script](#3-sft-training-script)
  - [4. Math Baseline Evaluation](#4-math-baseline-evaluation)
  - [5. Math Grading & Reward Functions](#5-math-grading--reward-functions)
  - [6. Prompt Templates](#6-prompt-templates)
  - [7. Generation Logging & Diagnostics](#7-generation-logging--diagnostics)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Prepare SFT Data](#prepare-sft-data)
  - [Run Math Baseline Evaluation](#run-math-baseline-evaluation)
  - [Run SFT Training](#run-sft-training)
  - [Run Tests](#run-tests)
- [Next Steps: GRPO / RL Training](#next-steps-grpo--rl-training)

---

## What's Been Implemented

### 1. SFT Data Preparation

**File:** [`cs336_alignment/prepare_sft_data.py`](./cs336_alignment/prepare_sft_data.py)

A complete data pipeline that downloads and prepares supervised fine-tuning data from the HuggingFace dataset [`eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1`](https://huggingface.co/datasets/eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1).

The pipeline applies three filtering/transformation steps:
1. **Tag normalization** — Swaps `<response>`/`</response>` tags to `<answer>`/`</answer>` and normalizes whitespace between `</think>` and `<answer>` tags to match the R1-Zero format.
2. **Format filtering** — Keeps only examples with correctly structured chain-of-thought: `<think>...</think> <answer>...</answer>`.
3. **Answer correctness filtering** — Validates that the answer extracted from the chain-of-thought matches the GSM8K ground-truth answer using the math grader. Also cleans up GSM8K-style `####` prefixes inside answer tags.

Outputs a `.jsonl` file with `{"prompt": ..., "response": ...}` records ready for SFT training.

### 2. SFT Core Utilities

**File:** [`cs336_alignment/sft.py`](./cs336_alignment/sft.py)

Core building blocks for supervised fine-tuning:

| Function | Description |
|---|---|
| `tokenize_prompt_and_output()` | Tokenizes prompt + response strings into `input_ids`, `labels` (shifted by 1), and `response_mask` (1 for response tokens, 0 for prompt/padding). Handles BOS tokens, padding, and batch construction. |
| `get_response_log_probs()` | Runs a forward pass through a causal LM and extracts per-token conditional log-probabilities via `log_softmax` + `gather`. Optionally returns per-token entropy. |
| `compute_entropy()` | Computes per-token entropy `H(p) = -Σ p(v) log p(v)` from logits using numerically stable `log_softmax`. |
| `masked_normalize()` | Sums tensor elements where `mask == 1` and divides by a normalization constant. Supports summing along a specific dimension or all dimensions. |
| `sft_microbatch_train_step()` | Computes the SFT cross-entropy loss (negative log-likelihood on response tokens only), scales for gradient accumulation, and calls `.backward()`. Returns loss and metadata (mean log-prob, perplexity, token count). |

### 3. SFT Training Script

**File:** [`cs336_alignment/train_sft.py`](./cs336_alignment/train_sft.py)

A full end-to-end SFT training loop for Qwen 2.5 Math 1.5B with the following features:

- **Optimizer:** AdamW with configurable learning rate and weight decay
- **LR Scheduler:** Cosine annealing with 10% minimum LR ratio
- **Gradient accumulation:** Supports microbatching (e.g., microbatch size 8 → effective batch size 32)
- **Gradient clipping:** Max gradient norm = 1.0
- **Periodic vLLM evaluation:** Loads the current policy weights into a vLLM instance on a separate GPU and evaluates on GSM8K validation examples with the R1-Zero reward function
- **WandB integration:** Logs training loss, gradient norm, LR, and evaluation accuracy/rewards on separate x-axes (`train_step` / `eval_step`)
- **Checkpoint saving:** Saves model + tokenizer after each epoch
- **Dataset size ablation:** Supports training on {128, 256, 512, 1024, full} examples via `--n_train_examples`
- **Dual-GPU setup:** Training on one GPU, vLLM evaluation on another

### 4. Math Baseline Evaluation

**File:** [`cs336_alignment/math_baseline.py`](./cs336_alignment/math_baseline.py)

Zero-shot evaluation of Qwen 2.5 Math 1.5B on GSM8K using the R1-Zero prompt template. Key components:

| Function | Description |
|---|---|
| `load_gsm8k_examples()` | Loads GSM8K JSONL data and extracts ground-truth answers from the `#### <number>` format. |
| `format_prompts()` | Formats questions using the R1-Zero prompt template with `<think>` continuation. |
| `evaluate_vllm()` | Generates responses with vLLM, computes rewards (format + answer correctness), and serializes per-example results and aggregate metrics to disk. Reports accuracy, format reward, and answer reward. |

### 5. Math Grading & Reward Functions

**File:** [`cs336_alignment/drgrpo_grader.py`](./cs336_alignment/drgrpo_grader.py)

A comprehensive math answer grading system (adapted from [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)) with multiple grading strategies for high recall:

- **`grade()`** — Main grading function combining multiple methods:
  - `grade_answer_mathd()` — Normalized string matching (from MATH dataset)
  - `grade_answer_sympy()` — Symbolic equivalence checking via SymPy
  - `is_latex_equal()` — LaTeX-aware comparison using `math_verify` (optional slow mode)
- **`r1_zero_reward_fn()`** — Reward function for R1-Zero formatted responses: checks for `</think> <answer>...</answer>` format compliance and answer correctness. Returns `{format_reward, answer_reward, reward}`.
- **`question_only_reward_fn()`** — Simpler reward function for `\boxed{}` formatted answers.
- **`extract_answer()`** — Extracts content from `\boxed{...}` LaTeX commands.
- **Robust normalization** — Handles units, LaTeX, fractions, mixed numbers, commas, percentages, and more.

### 6. Prompt Templates

**Directory:** [`cs336_alignment/prompts/`](./cs336_alignment/prompts/)

| Template | Description |
|---|---|
| `r1_zero.prompt` | R1-Zero style: instructs the model to think inside `<think>` tags and answer inside `<answer>` tags. Used for SFT and GRPO training/evaluation. |
| `alpaca_sft.prompt` | Alpaca instruction-following format with `### Instruction:` / `### Response:` structure. |
| `question_only.prompt` | Bare question text with no framing (for direct `\boxed{}` style answers). |
| `zero_shot_system_prompt.prompt` | Safety-aware system prompt for general-purpose instruction following. |

### 7. Generation Logging & Diagnostics

**Function:** `log_generations()` in [`cs336_alignment/sft.py`](./cs336_alignment/sft.py)

Rich generation logging utility used during training to monitor model behavior:
- Generates responses via vLLM for a sample of prompts
- Logs per-example: prompt, response, ground truth, rewards (format/answer/total), response length (tokens), and mean token entropy
- Computes aggregate statistics: mean/std response length, mean length for correct vs. incorrect responses, mean rewards

---

## Repository Structure

```
assignment5-alignment/
├── cs336_alignment/
│   ├── __init__.py
│   ├── prepare_sft_data.py       # SFT data download, filtering & preparation
│   ├── sft.py                    # SFT utilities (tokenization, loss, log-probs)
│   ├── train_sft.py              # Full SFT training loop with vLLM eval
│   ├── math_baseline.py          # Zero-shot GSM8K baseline evaluation
│   ├── drgrpo_grader.py          # Math grading & reward functions
│   └── prompts/
│       ├── r1_zero.prompt        # R1-Zero think/answer prompt
│       ├── alpaca_sft.prompt     # Alpaca instruction-following prompt
│       ├── question_only.prompt  # Plain question prompt
│       └── zero_shot_system_prompt.prompt
├── tests/
│   ├── adapters.py               # Test adapter layer (connects impl ↔ tests)
│   ├── test_sft.py               # SFT unit tests (all passing ✅)
│   ├── test_grpo.py              # GRPO unit tests (not yet implemented ❌)
│   ├── test_dpo.py               # DPO unit tests (optional, not implemented)
│   ├── test_data.py              # Data preparation tests
│   ├── test_metrics.py           # Metric tests
│   ├── conftest.py               # Pytest fixtures
│   ├── fixtures/                 # Test data (tokenizers, models, sample data)
│   └── _snapshots/               # Expected test outputs (numpy snapshots)
├── scripts/
│   ├── evaluate_safety.py        # Safety evaluation script
│   └── alpaca_eval_vllm_llama3_3_70b_fn/  # AlpacaEval config
├── data/                         # Dataset directory
├── pyproject.toml                # Project config (uv/pip dependencies)
├── CHANGELOG.md                  # Assignment version history
└── cs336_spring2025_assignment5_alignment.pdf  # Assignment handout
```

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. Install all packages (flash-attn requires special handling)
uv sync --no-install-package flash-attn
uv sync

# 2. Verify installation by running tests
uv run pytest tests/test_sft.py -v
```

### Dependencies

Key libraries: `torch`, `transformers`, `vllm` (v0.7.2), `flash-attn`, `wandb`, `math-verify`, `alpaca-eval`, `accelerate`

---

## Usage

### Prepare SFT Data

Download the GSM8K-enhanced dataset and filter it for high-quality SFT examples:

```bash
uv run python -m cs336_alignment.prepare_sft_data \
    --output_path data/sft_train.jsonl \
    --split train
```

### Run Math Baseline Evaluation

Evaluate the base Qwen 2.5 Math 1.5B model zero-shot on GSM8K:

```bash
uv run python -m cs336_alignment.math_baseline \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --data_path data/gsm8k/test.jsonl \
    --output_path outputs/math_baseline_results.jsonl \
    --top_n 30 \
    --tensor_parallel_size 4
```

### Run SFT Training

Train with the full dataset (requires 2 GPUs — one for training, one for vLLM evaluation):

```bash
uv run python -m cs336_alignment.train_sft \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --train_data data/sft_train.jsonl \
    --val_data data/gsm8k/test.jsonl \
    --output_dir outputs/sft_full \
    --n_train_examples -1 \
    --train_gpu 0 --vllm_gpu 1 \
    --wandb_project cs336-sft
```

Run a dataset size ablation (e.g., 128 examples):

```bash
uv run python -m cs336_alignment.train_sft \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --train_data data/sft_train.jsonl \
    --val_data data/gsm8k/test.jsonl \
    --output_dir outputs/sft_128 \
    --n_train_examples 128 \
    --train_gpu 0 --vllm_gpu 1 \
    --wandb_project cs336-sft
```

### Run Tests

```bash
# Run all SFT tests (should pass)
uv run pytest tests/test_sft.py -v

# Run all tests (GRPO tests will fail with NotImplementedError)
uv run pytest -v
```

---

## Next Steps: GRPO / RL Training

The next major milestone is implementing **GRPO (Group Relative Policy Optimization)** for reinforcement learning-based fine-tuning with verified math rewards. This involves implementing the following components:

### Functions to Implement

All stubs are in [`tests/adapters.py`](./tests/adapters.py) and the corresponding implementations should be added to the `cs336_alignment/` package:

1. **`compute_group_normalized_rewards()`**
   - Compute rewards for each group of rollout responses using `r1_zero_reward_fn`
   - Normalize rewards within each group (subtract group mean, optionally divide by group std)
   - Returns normalized advantages and raw rewards with metadata

2. **`masked_mean()`**
   - Compute the mean of a tensor along a dimension, considering only elements where `mask == 1`
   - Used throughout the GRPO loss computation for averaging over response tokens

3. **`compute_naive_policy_gradient_loss()`**
   - Vanilla REINFORCE: `-reward * log_prob` per-token loss
   - No baseline subtraction

4. **`compute_grpo_clip_loss()`**
   - PPO-style clipped objective adapted for GRPO
   - Clips the importance ratio `π_θ / π_old` to `[1 - ε, 1 + ε]`
   - Returns per-token loss and metadata (for computing clip fraction)

5. **`compute_policy_gradient_loss()`**
   - Dispatcher that delegates to the appropriate loss function based on `loss_type`:
     - `"no_baseline"` → naive policy gradient
     - `"reinforce_with_baseline"` → REINFORCE with advantage baseline
     - `"grpo_clip"` → GRPO clipped loss

6. **`grpo_microbatch_train_step()`**
   - GRPO equivalent of `sft_microbatch_train_step()`
   - Compute policy gradient loss, apply response masking, scale for gradient accumulation, call `.backward()`

### GRPO Training Loop

After implementing the above primitives, a full GRPO training script (similar to `train_sft.py`) should be created with:

- **Rollout generation:** Generate multiple responses per prompt using vLLM (group size G)
- **Reward computation:** Score responses with `r1_zero_reward_fn` and compute group-normalized advantages
- **Policy update:** Compute GRPO clipped loss, accumulate gradients across microbatches, and update the policy
- **KL penalty / entropy bonus:** Optionally add KL divergence regularization against the reference policy and/or entropy bonus to prevent mode collapse
- **Online iteration:** Alternate between rollout generation and policy updates across training steps

### Reference Papers

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948)
- [Dr. GRPO: Understanding GRPO and R1-Zero-like Training](https://github.com/sail-sg/understand-r1-zero)

### Validation

All GRPO test cases with expected snapshot outputs are already provided in `tests/test_grpo.py` and `tests/_snapshots/`. Once implemented, run:

```bash
uv run pytest tests/test_grpo.py -v
```
