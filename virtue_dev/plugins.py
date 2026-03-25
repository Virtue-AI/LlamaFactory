"""LiteFT plugin registrations for LlamaFactory v1.

Import this module before calling run_sft() to register all plugins.

    import virtue_dev.plugins  # noqa: F401  -- registers plugins as side effect
"""

from __future__ import annotations

import sys
from types import MethodType

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


# ── Liger kernel patches (pre-load) ─────────────────────────────────


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


def apply_fused_moe_patches(fused_moe_path: str | None = None):
    """Apply fused MoE grouped GEMM patches. Must be called BEFORE model loading.

    Requires the transformers-qwen3-moe-fused repo.
    """
    if fused_moe_path is not None:
        sys.path.insert(0, fused_moe_path)

    from qwen3_moe_fused.modular_qwen3_moe_fused import patch_Qwen3MoeSparseMoeBlock_init
    from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
    from qwen3_moe_fused.lora import patch_lora_config

    patch_Qwen3MoeSparseMoeBlock_init()
    patch_lora_config()
    patch_Qwen3MoeFusedSparseMoeBlock_forward()


# ── Post-load model patches ─────────────────────────────────────────

# Parameters that Liger's lce_forward accepts (everything else is FA3 kwargs)
_LCE_FORWARD_PARAMS = {
    "input_ids", "attention_mask", "position_ids", "past_key_values",
    "inputs_embeds", "labels", "use_cache", "output_attentions",
    "output_hidden_states", "output_router_logits", "cache_position",
    "logits_to_keep", "skip_logits",
}


def patch_lce_forward(model):
    """Patch model.forward so Liger's fused-linear CE works with FA3 kwargs.

    Liger's lce_forward passes **kwargs to both model() and the loss function.
    FA3 kwargs (cu_seq_lens_*, max_length_*) must reach model() but NOT the
    loss function. This wrapper separates them.

    Must be called AFTER model loading.
    """
    from liger_kernel.transformers.model.qwen3_moe import lce_forward

    def _filtered_forward(self, **kwargs):
        clean = {k: v for k, v in kwargs.items() if k in _LCE_FORWARD_PARAMS}
        extra = {k: v for k, v in kwargs.items() if k not in _LCE_FORWARD_PARAMS}
        orig = self.model.forward

        def inject_fa3(*a, **kw):
            kw.update(extra)
            return orig(*a, **kw)

        self.model.forward = inject_fa3
        try:
            return lce_forward(self, **clean)
        finally:
            self.model.forward = orig

    model.forward = MethodType(_filtered_forward, model)


def apply_neftune(model, alpha: float = 5.0):
    """Add NEFTune uniform noise to embeddings during training.

    Must be called AFTER model loading.
    """
    def _neftune_hook(module, input, output):
        if module.training:
            mag = alpha / (output.shape[-1] * output.shape[-2]) ** 0.5
            output = output + output.new_empty(output.shape).uniform_(-mag, mag)
        return output

    embed = model.get_input_embeddings()
    embed.register_forward_hook(_neftune_hook)


# ── Momentum scheduling ─────────────────────────────────────────────


class MomentumScheduler:
    """Wraps an LR scheduler and adds Muon momentum warmup/cooldown.

    Piggybacks on lr_scheduler.step() so it works with v1 BaseTrainer
    without modifying the training loop.

    Config: momentum_min (0.85), momentum_max (0.95),
            warmup_frac (0.05), cooldown_steps (50).
    """

    def __init__(self, lr_scheduler, optimizer, num_training_steps, config):
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self._step = 0

        self.momentum_min = config.get("momentum_min", 0.85)
        self.momentum_max = config.get("momentum_max", 0.95)
        warmup_frac = config.get("warmup_frac", 0.05)
        cooldown_steps = config.get("cooldown_steps", 50)

        self._warmup_steps = int(num_training_steps * warmup_frac)
        self._cooldown_start = max(0, num_training_steps - cooldown_steps)
        self._total_steps = num_training_steps

    def step(self):
        self.lr_scheduler.step()
        self._step += 1
        mom = self._get_momentum(self._step)
        for group in self.optimizer.param_groups:
            if "momentum" in group:
                group["momentum"] = mom

    def _get_momentum(self, step):
        if step < self._warmup_steps:
            frac = step / max(self._warmup_steps, 1)
            return self.momentum_min + frac * (self.momentum_max - self.momentum_min)
        if step >= self._cooldown_start:
            frac = (step - self._cooldown_start) / max(self._total_steps - self._cooldown_start, 1)
            frac = min(frac, 1.0)
            return self.momentum_max - frac * (self.momentum_max - self.momentum_min)
        return self.momentum_max

    def __getattr__(self, name):
        return getattr(self.lr_scheduler, name)


# ── VarlenPackingCollator ────────────────────────────────────────────


class VarlenPackingCollator:
    """Pack variable-length sequences into fixed-size tensors with cu_seqlens
    for Flash Attention varlen kernels.

    Each batch is packed into shape (1, target_length) with:
    - cu_seqlens: cumulative sequence lengths for FA varlen
    - position_ids: per-sequence position indices
    """

    def __init__(self, target_length: int, pad_token_id: int = 0):
        self.target_length = target_length
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        packed_ids: list[int] = []
        packed_labels: list[int] = []
        seq_lengths: list[int] = []
        current_length = 0

        for sample in batch:
            ids = sample["input_ids"]
            seq_len = len(ids)
            if current_length + seq_len > self.target_length:
                continue
            packed_ids.extend(ids)
            packed_labels.extend(sample["labels"])
            seq_lengths.append(seq_len)
            current_length += seq_len

        # Fallback: if nothing fit, truncate the first sample
        if not seq_lengths:
            ids = batch[0]["input_ids"][: self.target_length]
            labs = batch[0]["labels"][: self.target_length]
            packed_ids = list(ids)
            packed_labels = list(labs)
            seq_lengths = [len(ids)]
            current_length = len(ids)

        pad_len = self.target_length - current_length
        if pad_len > 0:
            packed_ids.extend([self.pad_token_id] * pad_len)
            packed_labels.extend([-100] * pad_len)
            seq_lengths.append(pad_len)

        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(seq_lengths, dtype=torch.int32), dim=0
        )
        position_ids = torch.cat(
            [torch.arange(sl, dtype=torch.long) for sl in seq_lengths]
        )

        return {
            "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),
            "labels": torch.tensor(packed_labels, dtype=torch.long).unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(seq_lengths),
            "num_packed": len(seq_lengths) - (1 if pad_len > 0 else 0),
            "pad_waste": pad_len,
        }
