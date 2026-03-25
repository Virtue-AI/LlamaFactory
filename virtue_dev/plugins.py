"""LiteFT plugin registrations for LlamaFactory v1.

Import this module before calling run_sft() to register all plugins.

    import virtue_dev.plugins  # noqa: F401  -- registers plugins as side effect
"""

import torch
from torch.optim.lr_scheduler import LambdaLR

from llamafactory.v1.plugins.trainer_plugins.optimizer import OptimizerPlugin
from llamafactory.v1.plugins.trainer_plugins.lr_scheduler import LRSchedulerPlugin


# ── Optimizers ───────────────────────────────────────────────────────


@OptimizerPlugin("muon").register()
def create_muon_optimizer(model, config):
    """Muon optimizer (requires `pip install muon`).

    Config: lr (2e-4), momentum (0.95), weight_decay (0.01).
    """
    from muon import SingleDeviceMuon

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return SingleDeviceMuon(
        trainable_params,
        lr=config.get("lr", 2e-4),
        momentum=config.get("momentum", 0.95),
        weight_decay=config.get("weight_decay", 0.01),
    )


@OptimizerPlugin("adamw").register()
def create_adamw_optimizer(model, config):
    """Standard AdamW optimizer.

    Config: lr (6e-5), weight_decay (0.01).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable_params,
        lr=config.get("lr", 6e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )


@OptimizerPlugin("adamw8bit").register()
def create_adamw8bit_optimizer(model, config):
    """8-bit AdamW optimizer (requires `pip install bitsandbytes`).

    Config: lr (6e-5), weight_decay (0.01).
    """
    import bitsandbytes as bnb

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.AdamW8bit(
        trainable_params,
        lr=config.get("lr", 6e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )


# ── LR Scheduler ────────────────────────────────────────────────────


@LRSchedulerPlugin("warmup_stable_cooldown").register()
def create_warmup_stable_cooldown_scheduler(optimizer, num_training_steps, config):
    """Warmup -> constant -> linear cooldown to a floor ratio.

    Config: warmup_ratio (0.05), cooldown_ratio (0.60), cooldown_floor (0.15).
    """
    warmup_ratio = config.get("warmup_ratio", 0.05)
    cooldown_ratio = config.get("cooldown_ratio", 0.60)
    cooldown_floor = config.get("cooldown_floor", 0.15)

    warmup_steps = int(num_training_steps * warmup_ratio)
    cooldown_start = int(num_training_steps * (1 - cooldown_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        if current_step >= cooldown_start:
            t = (current_step - cooldown_start) / max(num_training_steps - cooldown_start, 1)
            t = min(t, 1.0)
            return 1.0 * (1 - t) + cooldown_floor * t
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


# ── Liger kernel patches ────────────────────────────────────────────


def apply_liger_patches(
    rope: bool = True,
    rms_norm: bool = True,
    fused_linear_cross_entropy: bool = True,
    cross_entropy: bool = False,
    swiglu: bool = False,
):
    """Apply Liger fused kernels for Qwen3 MoE. Must be called BEFORE model loading."""
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

    apply_liger_kernel_to_qwen3_moe(
        rope=rope,
        rms_norm=rms_norm,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
        cross_entropy=cross_entropy,
        swiglu=swiglu,
    )
