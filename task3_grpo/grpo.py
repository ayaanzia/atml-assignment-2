from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Validate this implementation against the group-relative definition in the assignment manual.
    """
    if rewards.ndim != 1 or group_ids.ndim != 1 or rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must be matching one-dimensional tensors")
    if rewards.numel() == 0:
        raise ValueError("A GRPO batch needs at least one completion")
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("Group rewards must be finite")
    advantages = torch.empty_like(rewards)
    for group in torch.unique(group_ids):
        selected = group_ids == group
        values = rewards[selected]
        std = values.std(unbiased=False)
        # Sub-tolerance reward differences cannot supply a reliable ranking.
        advantages[selected] = (
            torch.zeros_like(values) if float(std.item()) <= eps
            else (values - values.mean()) / std
        )
    return advantages


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    if new_logp.shape != old_logp.shape or new_logp.shape != ref_logp.shape or new_logp.shape != token_mask.shape:
        raise ValueError("GRPO log-probability and token-mask shapes must match")
    if seq_adv.ndim != 1 or seq_adv.shape[0] != new_logp.shape[0]:
        raise ValueError("Need one advantage per completion")
    ratio = torch.exp(new_logp.float() - old_logp.float())
    adv = seq_adv.float()[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None or max_completion_length < 1:
            raise ValueError("dr_grpo requires a positive max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp.float() - new_logp.float()
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "per_sequence_policy_term": (-per_sequence).detach(),
        "kl_penalty": kl.detach(),
        "sampled_kl": kl.detach(),  # Legacy key: the objective uses a nonnegative KL estimator.
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
