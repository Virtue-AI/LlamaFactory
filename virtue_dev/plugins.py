"""LiteFT plugin registrations for LlamaFactory v1.

Import this module before calling run_sft() to register all plugins.

    import virtue_dev.plugins  # noqa: F401  -- registers plugins as side effect
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.optim.lr_scheduler import LambdaLR

from llamafactory.v1.plugins.trainer_plugins.optimizer import OptimizerPlugin
from llamafactory.v1.plugins.trainer_plugins.lr_scheduler import LRSchedulerPlugin
from llamafactory.v1.plugins.trainer_plugins.batching import BatchingPlugin


# ── Optimizers ───────────────────────────────────────────────────────


@OptimizerPlugin("muon").register()
def create_muon_optimizer(model: torch.nn.Module, config):
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
def create_adamw_optimizer(model: torch.nn.Module, config):
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
def create_adamw8bit_optimizer(model: torch.nn.Module, config):
    """8-bit AdamW optimizer (requires `pip install bitsandbytes`).

    Config: lr (6e-5), weight_decay (0.01).
    """
    import bitsandbytes as bnb

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.AdamW8bit(
        trainable_params,  # type: ignore[arg-type]
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


def apply_fused_moe_patches():
    """Apply fused MoE grouped GEMM patches. Must be called BEFORE model loading.

    Requires: uv pip install git+https://github.com/woct0rdho/transformers-qwen3-moe-fused
    """
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


# ── Padding-free batching plugin ─────────────────────────────────────

IGNORE_INDEX = -100


def _pack_samples(samples: list[dict], cutoff_len: int) -> dict:
    """Pack variable-length samples into a single sequence with attention_mask
    encoding for Flash Attention varlen.

    Each sequence gets a unique integer ID in attention_mask so that
    transformers' FA integration can compute cu_seqlens internally.
    """
    packed_ids: list[int] = []
    packed_labels: list[int] = []
    packed_weights: list[float] = []
    seq_lengths: list[int] = []
    current_length = 0
    seq_id = 1

    for sample in samples:
        ids = sample["input_ids"]
        seq_len = len(ids) if isinstance(ids, list) else ids.shape[-1]
        if current_length + seq_len > cutoff_len:
            continue
        if isinstance(ids, list):
            packed_ids.extend(ids)
            packed_labels.extend(sample.get("labels", [IGNORE_INDEX] * seq_len))
            packed_weights.extend(sample.get("loss_weights", [1.0] * seq_len))
        else:
            packed_ids.extend(ids.tolist())
            packed_labels.extend(sample.get("labels", torch.full((seq_len,), IGNORE_INDEX)).tolist())
            packed_weights.extend(sample.get("loss_weights", torch.ones(seq_len)).tolist())
        seq_lengths.append(seq_len)
        current_length += seq_len
        seq_id += 1

    # Fallback: if nothing fit, truncate the first sample
    if not seq_lengths:
        ids = samples[0]["input_ids"][:cutoff_len]
        labs = samples[0].get("labels", [IGNORE_INDEX] * len(ids))[:cutoff_len]
        weights = samples[0].get("loss_weights", [1.0] * len(ids))[:cutoff_len]
        if not isinstance(ids, list):
            ids, labs, weights = ids.tolist(), labs.tolist(), weights.tolist()
        packed_ids = list(ids)
        packed_labels = list(labs)
        packed_weights = list(weights)
        seq_lengths = [len(ids)]
        current_length = len(ids)

    # Pad to cutoff_len
    pad_len = cutoff_len - current_length
    if pad_len > 0:
        packed_ids.extend([0] * pad_len)
        packed_labels.extend([IGNORE_INDEX] * pad_len)
        packed_weights.extend([0.0] * pad_len)

    # Build attention_mask: unique ID per sequence, 0 for padding
    attention_mask: list[int] = []
    for i, sl in enumerate(seq_lengths):
        attention_mask.extend([i + 1] * sl)
    if pad_len > 0:
        attention_mask.extend([0] * pad_len)

    # Build position_ids: per-sequence positions
    position_ids: list[int] = []
    for sl in seq_lengths:
        position_ids.extend(range(sl))
    if pad_len > 0:
        position_ids.extend([0] * pad_len)

    return {
        "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),
        "labels": torch.tensor(packed_labels, dtype=torch.long).unsqueeze(0),
        "loss_weights": torch.tensor(packed_weights, dtype=torch.float).unsqueeze(0),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int32).unsqueeze(0),
        "position_ids": torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
    }


@BatchingPlugin("padding_free").register("compute_length")
def padding_free_compute_length(data_provider):
    """Estimate number of batches — same as normal since we pack on the fly."""
    return len(data_provider)


@BatchingPlugin("padding_free").register("fill_buffer")
def padding_free_fill_buffer(buffer, batch_info):
    """Pull samples into the buffer until we have enough to pack."""
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    # Over-fetch to increase packing density
    target_count = micro_batch_size * num_micro_batch
    while len(buffer) < target_count:
        try:
            samples = next(batch_info["data_iter"])
        except StopIteration:
            break
        buffer.put(samples)


@BatchingPlugin("padding_free").register("generate_batch")
def padding_free_generate_batch(buffer, batch_info):
    """Pack buffer samples into varlen micro-batches."""
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    cutoff_len = batch_info["cutoff_len"]
    total_needed = micro_batch_size * num_micro_batch

    if len(buffer) < total_needed:
        return None

    samples = buffer.get(total_needed)

    # Each micro-batch packs micro_batch_size samples into a single sequence
    batch = []
    for i in range(num_micro_batch):
        micro_samples = samples[i * micro_batch_size : (i + 1) * micro_batch_size]
        batch.append(_pack_samples(micro_samples, cutoff_len))

    return batch
