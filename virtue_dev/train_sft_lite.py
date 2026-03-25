"""SFT training with Liger kernels only (no fused MoE / NEFTune).

Usage:
    python virtue_dev/train_sft_lite.py config.yaml
"""

import virtue_dev.plugins as plugins  # noqa: F401

plugins.apply_liger_patches()

from llamafactory.v1.trainers.sft_trainer import run_sft

if __name__ == "__main__":
    run_sft()
