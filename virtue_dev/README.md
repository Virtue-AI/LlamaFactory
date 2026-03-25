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

- `plugins.py` — Registers optimizers (Muon, AdamW, AdamW8bit), LR scheduler (warmup-stable-cooldown), and Liger kernel patches via the v1 plugin system.
- `train_sft.py` — Entry point that loads plugins, applies Liger patches, and runs SFT training.
- `example_sft.yaml` — Example config using Muon + warmup-stable-cooldown + LoRA.

## Usage

```bash
python virtue_dev/train_sft.py virtue_dev/example_sft.yaml
```

Override config values from CLI:

```bash
python virtue_dev/train_sft.py virtue_dev/example_sft.yaml optim_config.lr=3e-4 num_train_epochs=3
```

## How it works

1. `import virtue_dev.plugins` registers optimizer/scheduler plugins into LlamaFactory's v1 global registry via decorators.
2. `apply_liger_patches()` monkey-patches transformers module classes (fused RoPE, RMSNorm, linear cross-entropy) before any model is loaded.
3. `run_sft()` reads the YAML config, resolves `optim_config.name` and `lr_scheduler_config.name` from the registry, and runs training as usual.

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
