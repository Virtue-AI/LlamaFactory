# virtue_dev

Custom training plugins for LlamaFactory v1, without modifying upstream source code.

## Install

```bash
git clone git@github.com:Virtue-AI/LlamaFactory.git
cd LlamaFactory
uv venv --python=3.12 --seed
source .venv/bin/activate
uv pip install -e .
uv pip install -r virtue_dev/requirements.txt
```

## What's inside

- `plugins.py` — All custom components:
  - **Optimizers**: Muon, AdamW, AdamW8bit
  - **LR Scheduler**: warmup-stable-cooldown
  - **Pre-load patches**: Liger kernels (fused RoPE/RMSNorm/Linear CE), fused MoE grouped GEMM
  - **Post-load patches**: LCE forward (FA3 kwargs separation), NEFTune (embedding noise)
  - **Momentum scheduling**: Muon momentum warmup/cooldown
  - **VarlenPackingCollator**: FA3 varlen sequence packing
- `train_sft.py` — Full entry point: Liger + fused MoE + LCE patch + NEFTune + Muon momentum scheduling.
- `train_sft_lite.py` — Lite entry point: Liger kernels + custom optimizer/scheduler only.
- `example_sft.yaml` — Example config using Muon + warmup-stable-cooldown + LoRA.

## Usage

Full pipeline (Qwen3 MoE with fused kernels + NEFTune):

```bash
python virtue_dev/train_sft.py virtue_dev/example_sft.yaml
```

Lite mode (Liger kernels only, works with any model):

```bash
python virtue_dev/train_sft_lite.py virtue_dev/example_sft.yaml
```

Override config values from CLI:

```bash
python virtue_dev/train_sft.py virtue_dev/example_sft.yaml optim_config.lr=3e-4 num_train_epochs=3
```

## How it works

1. `import virtue_dev.plugins` registers optimizer/scheduler plugins into LlamaFactory's v1 global registry via decorators.
2. Pre-load patches (`apply_liger_patches()`, `apply_fused_moe_patches()`) monkey-patch transformers module classes before any model is loaded.
3. Post-load patches (`patch_lce_forward()`, `apply_neftune()`) are applied to the loaded model instance.
4. `MomentumScheduler` wraps the LR scheduler to add Muon momentum warmup/cooldown without modifying the training loop.

## Adding new components

Add a new optimizer, scheduler, or loss function by registering it in `plugins.py`:

```python
@OptimizerPlugin("my_optimizer").register()
def create_my_optimizer(model, config):
    ...
```

Then reference it in your YAML config:

```yaml
optim_config:
  name: my_optimizer
  lr: 1e-4
```
