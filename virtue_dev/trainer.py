"""Custom SFT trainer aligned with LiteFT training behavior.

Differences from v1 BaseTrainer:
    - Grad clipping only for AdamW (Muon skips clip_grad_norm_)
    - Loss scaled by 1/grad_accum_steps (not by valid token ratio)
    - zero_grad(set_to_none=True) for lower memory usage
    - MomentumScheduler auto-wrapped for Muon optimizer
"""

from __future__ import annotations

import torch

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.trainers.sft_trainer import SFTTrainer
from llamafactory.v1.utils import logging

from . import plugins

logger = logging.get_logger(__name__)


class LiteFTTrainer(SFTTrainer):
    """SFTTrainer with LiteFT-aligned training loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wrap scheduler with momentum scheduling for Muon
        if self.args.optim_config and self.args.optim_config.name == "muon":
            self.lr_scheduler = plugins.MomentumScheduler(
                lr_scheduler=self.lr_scheduler,
                optimizer=self.optimizer,
                num_training_steps=self.num_training_steps,
                config=self.args.optim_config,
            )

    @property
    def _is_muon(self) -> bool:
        return self.args.optim_config is not None and self.args.optim_config.name == "muon"

    def fit(self) -> None:
        """Train the model with LiteFT-aligned behavior."""
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            self.train_batch_generator.set_epoch(epoch)
            for micro_batches in self.train_batch_generator:
                self.global_step += 1
                step_loss = 0
                num_micro = len(micro_batches)

                for i, micro_batch in enumerate(micro_batches):
                    loss = self.compute_loss(micro_batch)
                    # LiteFT: simple 1/N scaling instead of valid-token-weighted scaling
                    loss = loss / num_micro

                    if self._deepspeed_engine is not None:
                        self._deepspeed_engine.accelerator.sync_gradients = i == num_micro - 1
                        self._deepspeed_engine.backward(loss)
                    else:
                        loss.backward()
                    step_loss += loss.item()

                if self._deepspeed_engine is not None:
                    grad_norm = self._deepspeed_engine.get_grad_norm()
                else:
                    # LiteFT: only clip gradients for AdamW, not Muon
                    if not self._is_muon:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        ).item()

                        if not torch.isfinite(torch.tensor(grad_norm)):
                            logger.warning_rank0(f"Gradient norm is not finite: {grad_norm}")
                            self.lr_scheduler.step()
                            # LiteFT: set_to_none=True for lower memory
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                    else:
                        grad_norm = 0.0

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    # LiteFT: set_to_none=True for lower memory
                    self.optimizer.zero_grad(set_to_none=True)

                step_loss, grad_norm = DistributedInterface().all_reduce([step_loss, grad_norm])
                DistributedInterface().sync()
                if DistributedInterface().get_rank() == 0:
                    print(f"Epoch {epoch}, Step {self.global_step}, Loss: {step_loss:.4f}, Grad Norm: {grad_norm:.4f}")

                if self.global_step >= self.num_training_steps:
                    logger.info_rank0(f"Reached max_steps ({self.num_training_steps}), stopping training.")
                    return
