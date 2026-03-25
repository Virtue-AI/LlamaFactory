"""SFT training with LiteFT plugins (Liger kernels, Muon, custom scheduler).

Usage:
    python virtue_dev/train_sft.py config.yaml
    python virtue_dev/train_sft.py config.yaml optim_config.lr=3e-4

Liger patches are applied before model loading. Optimizer and scheduler
plugins are registered at import time via virtue_dev.plugins.
"""

# 1. Register optimizer / scheduler plugins (side effect on import)
import virtue_dev.plugins as plugins  # noqa: F401

# 2. Apply Liger kernel patches BEFORE any model is loaded
plugins.apply_liger_patches()

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config import InputArgument, get_args
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.trainers.sft_trainer import SFTTrainer


def run_sft(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args.dist_config)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)

    # ── Post-load patches ──
    # Patch Liger LCE forward to separate FA3 kwargs (only needed with FA3 + Liger LCE)
    # plugins.patch_lce_forward(model_engine.model)

    # NEFTune: uncomment and set alpha to enable
    # plugins.apply_neftune(model_engine.model, alpha=5.0)

    trainer = SFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )

    # ── Wrap scheduler with momentum scheduling for Muon ──
    if training_args.optim_config and training_args.optim_config.name == "muon":
        trainer.lr_scheduler = plugins.MomentumScheduler(
            lr_scheduler=trainer.lr_scheduler,
            optimizer=trainer.optimizer,
            num_training_steps=trainer.num_training_steps,
            config=training_args.optim_config,
        )

    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    run_sft()
