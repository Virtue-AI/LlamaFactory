# virtue_dev

Custom training plugins for LlamaFactory v1.

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
  - **Batching**: padding-free varlen sequence packing via `BatchingPlugin`
- `trainer.py` — `LiteFTTrainer` that aligns training behavior with LiteFT (grad clipping, loss scaling, etc.)
- `train_sft.py` — Full entry point: Liger + fused MoE + LCE patch + NEFTune + Muon momentum scheduling.
- `train_sft_lite.py` — Lite entry point: Liger kernels + custom optimizer/scheduler only.
- `example_sft.yaml` — Example config using Muon + warmup-stable-cooldown + LoRA + padding-free packing.

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

## Padding-free packing

Set `batching_strategy: padding_free` in your YAML config to enable varlen sequence packing. Multiple samples are packed into a single sequence with unique attention mask IDs per sequence, so Flash Attention computes varlen attention without wasted padding.

`micro_batch_size` controls how many samples are packed into each sequence. Example:

```yaml
micro_batch_size: 4
cutoff_len: 8192
batching_strategy: padding_free
```

This packs 4 samples into one 8192-token sequence per micro-batch.

## How it works

1. `import virtue_dev.plugins` registers optimizer, scheduler, and batching plugins into LlamaFactory's v1 global registry via decorators.
2. Pre-load patches (`apply_liger_patches()`, `apply_fused_moe_patches()`) monkey-patch transformers module classes before any model is loaded.
3. Post-load patches (`patch_lce_forward()`, `apply_neftune()`) are applied to the loaded model instance.
4. `LiteFTTrainer` overrides the training loop to align with LiteFT behavior: Muon skips grad clipping, loss uses simple 1/N scaling, and `MomentumScheduler` handles Muon momentum warmup/cooldown.
5. `BatchingPlugin("padding_free")` packs variable-length sequences into fixed-size tensors with per-sequence attention mask IDs, enabling Flash Attention varlen kernels.

## Adding new components

Register a new optimizer, scheduler, or batching strategy in `plugins.py`:

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
