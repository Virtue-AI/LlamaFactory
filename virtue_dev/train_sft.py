"""SFT training with LiteFT plugins (Liger kernels, Muon, custom scheduler).

Usage:
    python virtue_dev/train_sft.py config.yaml
    python virtue_dev/train_sft.py config.yaml optim_config.lr=3e-4

Liger + fused MoE patches are applied before model loading. Optimizer and
scheduler plugins are registered at import time via virtue_dev.plugins.
"""

# 1. Register optimizer / scheduler plugins (side effect on import)
import virtue_dev.plugins as plugins  # noqa: F401

# 2. Apply kernel patches BEFORE any model is loaded
plugins.apply_liger_patches()
plugins.apply_fused_moe_patches()

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config import InputArgument, get_args
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.core.model_engine import ModelEngine

from virtue_dev.trainer import LiteFTTrainer


def run_sft(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args.dist_config)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)

    # ── Post-load patches ──
    plugins.patch_lce_forward(model_engine.model)
    plugins.apply_neftune(model_engine.model, alpha=5.0)

    trainer = LiteFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    run_sft()
