# Training Configuration Files

This folder contains YAML configuration files for different VAE training experiments. Each config represents a different hypothesis about what hyperparameters might work best.

---

## Understanding the Configurations

Each YAML file controls key aspects of the training process. The two most important parameters that vary between configs are:

**Latent Dimension (latent_dim):** How compressed the representation is
- 256D = 16:1 compression (4096 Gaussian params → 256 latent dims)
- 512D = 8:1 compression (4096 Gaussian params → 512 latent dims)
- Larger latent spaces can store more information but are less compressed

**Beta (β_max):** Controls the KL divergence penalty in the loss function
- β=1.0 is the "standard" VAE formulation
- Lower β (like 0.5 or 0.1) reduces regularization pressure, which can help prevent posterior collapse
- We anneal β from 0 to β_max over the first 50 epochs to let the model learn reconstruction first

---

## Available Configurations

| Config | Latent | β_max | What I was testing |
|--------|--------|-------|---------|
| `conv1d_base.yaml` | 256D | 1.0 | Original baseline - standard VAE |
| `conv1d_beta0.5.yaml` | 256D | 0.5 | Lower beta with smaller latent |
| `conv1d_512d.yaml` | 512D | 1.0 | Bigger latent space, standard beta |
| `conv1d_512d_beta0.5.yaml` | 512D | 0.5 | Combined: bigger latent + lower beta |
| `conv1d_512d_beta0.1.yaml` | 512D | 0.1 | Much lower beta (next experiment) |
| `conv1d_512d_beta0.yaml` | 512D | 0.0 | Pure autoencoder (no KL penalty) |
| `resnet_conv1d_512d.yaml` | 512D | 0.0 → custom | ResNet architecture with weighted loss |

---

## What I Learned From Training Experiments

| Config | Run | Epochs | Final KL | Final Recon | What happened |
|--------|-----|--------|----------|-------------|--------|
| `conv1d_base.yaml` | run_002 | 34 | 0.06 | 3.16 | KL collapsed - posterior collapsed to prior |
| `conv1d_512d_beta0.5.yaml` | run_003 | 42 | 0.07 | 3.16 | KL still collapsed despite lower β |
| `conv1d_512d_beta0.yaml` | run_006 | 25 | N/A | 0.1 (xy), 1.2 (rot) | Pure AE - found loss imbalance issue! |

**Key insight from run_006:** With β=0 (pure autoencoder), the model learned XY positions great (loss 0.1) but rotation and features got stuck (loss ~1.0-1.2). This revealed that equal loss weighting causes the optimizer to ignore low-channel parameters.

**The problem:** When you sum losses across different parameter types, the optimizer naturally focuses on parameters with more channels (XY has 2, features have 3) and ignores single-channel ones (rotation has 1). It's like trying to hear a whisper in a noisy room.

**The solution (run_007):** Normalize each parameter type independently before summing, so rotation loss and feature loss get equal weight in the gradient. Think of it as giving each parameter type its own microphone.

---

## What's Next?

**Next training run (run_007):**
- Config: `resnet_conv1d_512d.yaml` 
- Key change: Weighted loss implementation
- Goal: Get rotation loss from 1.2 → 0.6 and feature loss from 1.0 → 0.7
- Expected improvement: Model predictions should have variance matching ground truth (std ~1.0 instead of collapsed std ~0.05)

**The weighted loss formula:**
```
total_loss = loss_xy/2 + loss_scale/2 + loss_rot/1 + loss_feat/3
```

Each term is normalized by the number of channels, so the optimizer treats all parameter types fairly.

---

## Common Parameters (Shared Across All Configs)

These stay the same across experiments so we can isolate what's actually causing differences:

```yaml
batch_size: 32              # Limited by T4 GPU memory
epochs: 100                 # Target (early stopping if needed)
learning_rate: 0.001        # Adam default, works well
beta_warmup_epochs: 50      # Linear ramp from 0 to β_max
```

**Encoder architecture:**
```yaml
encoder_filters: [32, 64, 128]  # Progressive channel expansion
```

**Decoder architecture:**
```yaml
decoder_filters: [64, 32, 16]   # Progressive channel reduction
```

The ResNet config (`resnet_conv1d_512d.yaml`) uses deeper blocks with residual connections for better gradient flow, which is why it has more parameters (~4.8M vs ~1.6M).

---

## How to Use These Configs

**Running a training experiment:**

```bash
python scripts/train_conv1d_vae.py \
  --config configs/conv1d_512d_beta0.1.yaml \
  --output experiments/run_004
```

**What happens:**
1. The script loads the YAML config
2. Creates the model with specified architecture
3. Sets up data pipeline with normalization
4. Trains with the specified hyperparameters
5. Saves checkpoints and TensorBoard logs to the output directory

**Monitoring training:**
```bash
tensorboard --logdir experiments/run_004/logs
```

This lets you watch the loss curves, latent space statistics, and reconstruction quality in real-time.

---

## Tips for Creating New Configs

If you want to experiment with new hyperparameters:

1. **Copy an existing config** that's close to what you want
2. **Change one thing at a time** so you know what caused any differences
3. **Document your hypothesis** - why do you think this change will help?
4. **Use descriptive names** like `conv1d_512d_beta0.05_higher_lr.yaml`

**Good things to experiment with:**
- `latent_dim`: Try 384D, 768D, or other values
- `max_beta`: Anything from 0.0 to 2.0
- `learning_rate`: Try 5e-4, 2e-3, etc.
- `batch_size`: Larger if you have more GPU memory

**Things that probably won't help:**
- Changing the filter sizes dramatically (the current ones work well)
- Using a learning rate > 0.01 (too unstable)
- Setting beta_warmup_epochs < 20 (model needs time to learn reconstruction first)

---

**Last Updated:** November 15, 2025

For more details on the loss weighting issue and solution, see `experiments/loss_weighting_strategies.md`.
