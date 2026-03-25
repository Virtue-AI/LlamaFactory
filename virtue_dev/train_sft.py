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

# 3. Run the standard v1 SFT flow (uses registered plugins via config)
from llamafactory.v1.trainers.sft_trainer import run_sft

if __name__ == "__main__":
    run_sft()
