"""
SFT (Supervised Fine-Tuning) utilities.

Provides tokenization helpers used for both SFT training and GRPO rollout scoring.
"""

from __future__ import annotations

import math
import statistics
import torch
from transformers import PreTrainedTokenizerBase


def log_generations(
    vllm_model,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn,
    sampling_params,
    tokenizer=None,
    step: int = 0,
    log_fn=None,
) -> dict:
    """Generate responses for a list of prompts and log rich diagnostics.

    Logs per-example:
      1. Input prompt
      2. Generated response
      3. Ground-truth answer
      4. Reward info (format_reward, answer_reward, reward)
      5. Average token entropy of the response
      6. Response length (in tokens)

    Also computes aggregate statistics:
      - Mean/std response length overall
      - Mean response length for correct vs. incorrect responses
      - Mean rewards

    Args:
        vllm_model: vLLM LLM instance used for generation.
        prompts: list[str], the formatted prompt strings.
        ground_truths: list[str], ground-truth answers parallel to prompts.
        reward_fn: Callable[[str, str], dict[str, float]], scores responses.
        sampling_params: vLLM SamplingParams for generation.
        tokenizer: optional tokenizer; if provided, used to count response tokens.
        step: int, current training step (for logging context).
        log_fn: optional callable(str) for custom logging (defaults to print).

    Returns:
        dict with aggregate statistics (mean rewards, lengths, etc.)
    """
    if log_fn is None:
        log_fn = print

    outputs = vllm_model.generate(prompts, sampling_params)

    rewards_all: list[float] = []
    format_rewards_all: list[float] = []
    answer_rewards_all: list[float] = []
    lengths_all: list[int] = []
    lengths_correct: list[int] = []
    lengths_incorrect: list[int] = []
    mean_entropies_all: list[float] = []

    log_fn(f"\n{'='*70}")
    log_fn(f"  LOG GENERATIONS — step {step}  ({len(prompts)} examples)")
    log_fn(f"{'='*70}")

    for i, output in enumerate(outputs):
        prompt = output.prompt
        response = output.outputs[0].text
        gt = ground_truths[i]

        # ── Rewards ───────────────────────────────────────────────────────────
        reward_info = reward_fn(response, gt)
        fmt_r = reward_info.get("format_reward", 0.0)
        ans_r = reward_info.get("answer_reward", 0.0)
        total_r = reward_info.get("reward", 0.0)

        rewards_all.append(total_r)
        format_rewards_all.append(fmt_r)
        answer_rewards_all.append(ans_r)

        # ── Response length ───────────────────────────────────────────────────
        if tokenizer is not None:
            resp_tokens = tokenizer.encode(response, add_special_tokens=False)
            resp_len = len(resp_tokens)
        else:
            resp_len = len(response.split())

        lengths_all.append(resp_len)
        if ans_r == 1.0:
            lengths_correct.append(resp_len)
        else:
            lengths_incorrect.append(resp_len)

        # ── Token entropy (from vLLM logprobs if available) ───────────────────
        # vLLM returns logprobs as a list of dicts {token_id: Logprob(logprob=..., ...)}
        # We compute per-token entropy as -sum_v p(v)*log p(v) ≈ -logprob of chosen token
        # when only top-k logprobs are stored. Here we use the stored logprobs directly.
        mean_entropy: float | None = None
        token_logprobs = output.outputs[0].logprobs
        if token_logprobs:
            per_token_entropies = []
            for tok_lp_dict in token_logprobs:
                # tok_lp_dict: dict[int, Logprob]; each Logprob has .logprob attribute
                lps = [lp.logprob for lp in tok_lp_dict.values()]
                # Compute entropy from the stored top-k logprobs
                # (approximation: only covers the mass in top-k)
                probs = [math.exp(lp) for lp in lps]
                total_prob = sum(probs)
                if total_prob > 0:
                    # Normalize to form a valid distribution over the top-k tokens
                    norm_probs = [p / total_prob for p in probs]
                    ent = -sum(p * math.log(p + 1e-12) for p in norm_probs)
                    per_token_entropies.append(ent)
            if per_token_entropies:
                mean_entropy = sum(per_token_entropies) / len(per_token_entropies)
        mean_entropies_all.append(mean_entropy if mean_entropy is not None else float("nan"))

        # ── Per-example log ───────────────────────────────────────────────────
        entropy_str = f"{mean_entropy:.3f}" if mean_entropy is not None else "N/A"
        log_fn(f"\n--- Example {i+1}/{len(prompts)} ---")
        log_fn(f"PROMPT      : ...{prompt[-200:]}")   # last 200 chars
        log_fn(f"RESPONSE    : {response}")
        log_fn(f"GROUND TRUTH: {gt}")
        log_fn(f"REWARDS     : format={fmt_r:.1f}  answer={ans_r:.1f}  total={total_r:.1f}")
        log_fn(f"RESP LEN    : {resp_len} tokens | MEAN ENTROPY: {entropy_str}")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def safe_mean(lst: list) -> float:
        # Filter out nan values (only possible for float entries; ints are always valid)
        valid = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
        return sum(valid) / len(valid) if valid else float("nan")

    stats = {
        "step": step,
        "n_examples": len(prompts),
        "n_correct": len(lengths_correct),
        "n_incorrect": len(lengths_incorrect),
        "mean_reward": safe_mean(rewards_all),
        "mean_format_reward": safe_mean(format_rewards_all),
        "mean_answer_reward": safe_mean(answer_rewards_all),
        "mean_response_length": safe_mean(lengths_all),
        "std_response_length": (
            statistics.stdev(lengths_all) if len(lengths_all) > 1 else 0.0
        ),
        "mean_response_length_correct": safe_mean(lengths_correct),
        "mean_response_length_incorrect": safe_mean(lengths_incorrect),
        "mean_token_entropy": safe_mean(mean_entropies_all),
    }

    log_fn(f"\n{'─'*70}")
    log_fn(f"AGGREGATE STATS (step {step}):")
    for k, v in stats.items():
        if isinstance(v, float):
            log_fn(f"  {k}: {v:.4f}")
        else:
            log_fn(f"  {k}: {v}")
    log_fn(f"{'='*70}\n")

    return stats


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the per-token entropy of next-token predictions.

    Uses the numerically stable log-softmax formula:
        H(p) = -sum_v p(v) * log p(v)
             = -sum_v softmax(logits)_v * log_softmax(logits)_v

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            per-token entropy of the next-token distribution.
    """
    # log_softmax is numerically stable (uses logsumexp internally)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (..., vocab_size)
    probs = torch.exp(log_probs)                                  # (..., vocab_size)
    # H = -sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)                   # (batch, seq_len)
    return entropy


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch for SFT.

    Computes the SFT cross-entropy loss (negative log-likelihood on response
    tokens only), scales it for gradient accumulation, calls .backward(), and
    returns the loss value for logging.

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probs
            from the SFT policy being trained.
        response_mask: (batch_size, sequence_length) 1 for response tokens,
            0 for prompt/padding.
        gradient_accumulation_steps: number of microbatches per optimizer step.
            The loss is divided by this so that gradients from all microbatches
            sum to the equivalent of one full-batch gradient.
        normalize_constant: divide the masked sum by this constant before
            scaling for gradient accumulation.

    Returns:
        tuple[torch.Tensor, dict]:
            loss: scalar tensor (detached value for logging).
            metadata: dict with any extra statistics.
    """
    # SFT loss = - mean-over-batch of masked sum / normalize_constant
    # Equivalent to dividing by (batch_size * normalize_constant)
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(policy_log_probs, response_mask,
                             normalize_constant=normalize_constant * batch_size,
                             dim=None)

    # Scale for gradient accumulation so that accumulated gradients equal
    # the gradient of the full-batch loss.
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    # Useful metadata for monitoring training health
    n_response_tokens = response_mask.sum().item()
    mean_log_prob = (policy_log_probs * response_mask).sum().item() / max(n_response_tokens, 1)
    metadata: dict[str, float] = {
        # Average log-prob per response token (tracks training progress)
        "mean_response_log_prob": mean_log_prob,
        # Perplexity of the model on response tokens (lower = better fit)
        "response_perplexity": float(torch.exp(-torch.tensor(mean_log_prob)).item()),
        # Number of response tokens in this microbatch (useful for debugging padding)
        "n_response_tokens": n_response_tokens,
    }
    # Return scaled_loss (divided by gradient_accumulation_steps) so the
    # returned value matches what was actually backpropagated.
    return scaled_loss, metadata


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over tensor elements (respecting mask) and normalize by a constant.

    Only positions where mask == 1 contribute to the sum;
    masked positions (mask == 0) are zeroed out before summing.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, same shape as tensor. 1 = include, 0 = exclude.
        normalize_constant: float, divide the sum by this value.
        dim: int | None, dimension to sum along. If None, sum over all dims.

    Returns:
        torch.Tensor: the masked sum divided by normalize_constant.
    """
    masked = tensor * mask  # zero out excluded positions
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal language model,
    and optionally the per-token entropy of the next-token distribution.

    Args:
        model: HuggingFace causal LM. Should already be on the correct device
               and in eval/no_grad mode if gradients are not needed.
        input_ids: torch.Tensor of shape (batch_size, sequence_length)
            Concatenated prompt + response token ids (from tokenize_prompt_and_output).
        labels: torch.Tensor of shape (batch_size, sequence_length)
            Shifted input_ids (from tokenize_prompt_and_output), i.e., labels[t] = input_ids[t+1].
        return_token_entropy: bool
            If True, also compute and return per-token entropy.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": shape (batch_size, sequence_length)
                log p_theta(x_t | x_{<t}) for each position t.
            "token_entropy": shape (batch_size, sequence_length), optional
                Per-token entropy (present only if return_token_entropy=True).
    """
    # Forward pass — get logits of shape (batch_size, sequence_length, vocab_size)
    logits = model(input_ids).logits  # (B, T, V)

    # log_softmax over vocab dim → log p(x | context)
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather the log-prob of each actual next token (from labels)
    # labels shape: (B, T), values are token indices
    # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    # output[b][s][0] = log_probs_all[b][s][ index[b][s][0] ] --> same shape as index (B, T, 1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # log_probs: (B, T)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)  # (B, T)

    return result


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    The prompt and output are tokenized separately, then concatenated.
    Special tokens (BOS) are added only at the start of the prompt, not at
    the prompt/output boundary.

    Args:
        prompt_strs: list[str], the prompt strings (batch).
        output_strs: list[str], the output strings (batch).
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max_len - 1)
                Tokenized prompt+output concatenated, with the last token
                sliced off (used as the LM input).
            "labels": torch.Tensor of shape (batch_size, max_len - 1)
                Shifted token ids — the tokens without the first one
                (used as the LM target).
            "response_mask": torch.Tensor of shape (batch_size, max_len - 1)
                1 for response tokens in `labels`, 0 for prompt/padding tokens.
    """
    assert len(prompt_strs) == len(output_strs), (
        f"prompt_strs and output_strs must have the same length, "
        f"got {len(prompt_strs)} and {len(output_strs)}"
    )

    batch_size = len(prompt_strs)

    # Tokenize prompts — add BOS/special tokens at the start
    prompt_enc = tokenizer(
        prompt_strs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    # Tokenize outputs — no special tokens at the boundary
    output_enc = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    # Determine max combined length for padding
    combined_lens = [
        len(p) + len(o)
        for p, o in zip(prompt_enc["input_ids"], output_enc["input_ids"])
    ]
    max_len = max(combined_lens)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Allocate output tensors of shape (batch_size, max_len)
    # We fill padded and response_mask_full in a single pass per example.
    padded = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
    # response_mask_full[i, t] = 1 iff position t in the combined sequence is
    # a response token (i.e., t in [prompt_len, prompt_len + output_len))
    response_mask_full = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (p_ids, o_ids) in enumerate(
        zip(prompt_enc["input_ids"], output_enc["input_ids"])
    ):
        p_len = len(p_ids)
        o_len = len(o_ids)
        seq_len = p_len + o_len  # = combined_lens[i]

        # Fill token ids
        padded[i, :seq_len] = torch.tensor(p_ids + o_ids, dtype=torch.long)

        # Mark response positions in the full (unshifted) sequence
        # NOTE: only response tokens have mask value of 1 
        response_mask_full[i, p_len:seq_len] = 1

    # ── Shift to get input_ids / labels / response_mask ───────────────────────
    #
    # Causal LM convention:
    #   input_ids[t] = token t        → shape (batch_size, max_len - 1)
    #   labels[t]    = token t+1      → shape (batch_size, max_len - 1)
    #
    # response_mask on labels: labels[t] is a response token iff
    # position t+1 in the original sequence is a response token,
    # i.e., response_mask_full[:, 1:].

    input_ids     = padded[:, :-1]               # (B, max_len-1)
    labels        = padded[:, 1:]                # (B, max_len-1)
    response_mask = response_mask_full[:, 1:]    # (B, max_len-1)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
