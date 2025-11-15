# GaussVAE: Learning to Compress Images with 2D Gaussian Splatting

> A deep learning project exploring how Variational Autoencoders can learn compact representations of 2D Gaussian Splats for image compression and abstract art generation.

**Current Status:** Fine-tuning training hyperparameters to fix loss imbalance issues (November 2025)

---

## What is this project about?

This is a research project that combines two cool ideas:

1. **2D Gaussian Splatting** - A technique for representing images as a collection of colorful, fuzzy ellipses (think of them as soft paint splatters). Each "splat" has a position, size, rotation, and color.

2. **Variational Autoencoders (VAEs)** - Neural networks that can compress complex data into small latent codes, then reconstruct it back. Like a smart ZIP file that understands the structure of what it's compressing.

The twist? Instead of compressing pixels directly, we compress the *Gaussian representation* of images. This lets us manipulate images in interesting ways through their latent codes.

**The Pipeline:**
```
Image → 512 Gaussian Splats → 512D latent code → Reconstructed Splats → Image
```

This is the first time (as far as I know) that VAEs have been applied to learned 2D Gaussian representations. It's a fun intersection of generative models and novel rendering techniques!

---

## Current Progress

**What's working:**
- ✅ Full dataset of 11,501 abstract art images converted to Gaussian splats
- ✅ Complete VAE implementation (9 production files, 173 tests passing)
- ✅ Docker setup with GPU support for reproducible training
- ✅ Multi-image demo notebook showing reconstructions

**What I'm working on:**
- 🔄 Fixing loss weighting issues (some parameters weren't learning properly)
- 🔄 Training run #7 with improved per-channel normalization
- 🎯 Goal: Get rotation and color features to learn as well as position/scale

**Results so far:**
- 6 training runs completed with various hyperparameters
- Discovered that equal loss weighting causes some channels to collapse
- Next experiment will use weighted losses based on parameter types

---

## Getting Started

Want to play around with this? Here's what you'll need:

**Hardware:**
- An NVIDIA GPU (I've tested this on a Tesla T4, but any CUDA-compatible GPU should work)
- About 20GB of disk space

**Software:**
- Docker and docker-compose (for containerized training)
- NVIDIA drivers and nvidia-docker2 (so Docker can use your GPU)
- That's it! Everything else runs in containers

**Installation:**

```bash
git clone https://github.com/rokocuba/GaussVae-showcase.git
cd GaussVae-showcase
```

The Image-GS rendering library is already included in `third_party/image-gs/`, so no need to hunt down dependencies.

---

## How to Use

### 1. Build the Docker Images

We use two separate containers - one for converting images to Gaussians, and one for VAE training:

```bash
# Image-GS container (for encoding/decoding images)
docker build -f docker/Dockerfile.gaussvae-imagegs -t gaussvae-imagegs .

# VAE training container (TensorFlow + CUDA)
docker build -f docker/Dockerfile.gaussvae-vae -t gaussvae-vae .
```

### 2. Launch Jupyter Lab for Interactive Work

```bash
docker-compose up -d jupyter
```

Head to `http://localhost:8888` in your browser. If you're on a remote machine, VS Code's port forwarding works great too.

**Quick GPU check:**
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

**Interesting notebooks to explore:**
- `notebooks/08_vae_multi_image_demo.ipynb` - See VAE reconstructions on multiple images
- `notebooks/02_architecture_prototyping.ipynb` - Original VAE development notebook

### 3. Convert an Image to Gaussian Splats

This is where the magic starts - taking a regular image and finding the best set of Gaussian splats to represent it:

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/third_party/image-gs:/workspace/third_party/image-gs \
  -w /workspace/third_party/image-gs \
  gaussvae-imagegs \
  python main.py \
  --input_path /workspace/data/sample.png \
  --exp_name test/sample \
  --num_gaussians 500 \
  --max_steps 5000
```

**Output:** A checkpoint file at `third_party/image-gs/results/test/sample/.../checkpoints/ckpt_step-5000.pt`

Takes about 30 seconds on a Tesla T4 for a 256×256 image.

### 4. Render Gaussians Back to an Image

Once you have Gaussian splats, you can render them back to see how good the approximation is:

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/third_party/image-gs:/workspace/third_party/image-gs \
  -w /workspace/third_party/image-gs \
  gaussvae-imagegs \
  python main.py \
  --input_path /workspace/data/sample.png \
  --exp_name test/sample \
  --num_gaussians 500 \
  --eval \
  --render_height 256
```

**Output:** Rendered image at `third_party/image-gs/results/test/sample/.../eval/render_res-256x256.png`

### 5. Clean Up

```bash
docker-compose down
```

---

## The VAE Architecture

**What's under the hood:**
- **Type:** 1D Convolutional VAE (treats the 512 Gaussians as a sequence)
- **Size:** ~4.8M parameters (ResNet version with residual connections)
- **Compression:** 512 Gaussians × 8 params = 4096D → 512D latent space (8:1 compression)
- **Framework:** TensorFlow 2.12 with Keras

**The architecture uses:**
- ResNet-style encoder/decoder with residual blocks
- 1D convolutions to process the spatially-sorted Gaussians
- Standard VAE loss: reconstruction error + β×KL divergence

**Current challenge:**
The model was learning positions and scales really well, but rotation and color features were getting stuck. After some debugging, I found this was due to loss weighting issues - the optimizer was essentially ignoring single-channel parameters (rotation) in favor of multi-channel ones (XY positions, RGB colors).

The fix: normalize each parameter type independently before summing losses, so the optimizer treats all channels fairly.

**Train your own:**
```bash
python scripts/train_conv1d_vae.py \
  --config configs/resnet_conv1d_512d.yaml \
  --output experiments/my_run
```

For more technical details: `instructions/VAE_ARCHITECTURE.md`

---

## About the Data

The project uses a dataset of 11,501 abstract art images (from a Delaunay triangulation generator). Each image gets converted to 512 Gaussian splats, which are then stored efficiently.

**Data format:**

Originally stored as PyTorch `.pt` files (674 MB total), I converted everything to NumPy `.npz` format with compression, shrinking it down to 173 MB - a 74% reduction while keeping all the data lossless.

```python
import numpy as np

# Loading a sample is straightforward
data = np.load('data/delaunay/npz/train/img_000001.npz')
gaussians = data['gaussians']  # Shape: (512, 8) - 512 Gaussians with 8 parameters each

# Each Gaussian has 8 parameters:
xy = gaussians[:, 0:2]      # Position (x, y) in [0, 1]
scale = gaussians[:, 2:4]   # Ellipse width & height (in log space)
rot = gaussians[:, 4:5]     # Rotation angle [-π, π]
feat = gaussians[:, 5:8]    # RGB color [0, 1]

# Plus some quality metrics
psnr = float(data['psnr'])   # Peak Signal-to-Noise Ratio
ssim = float(data['ssim'])   # Structural Similarity Index
lpips = float(data['lpips']) # Perceptual similarity (lower is better)
```

**Key preprocessing steps:**
1. **Morton sorting** - Rearranges Gaussians following a Z-order curve for spatial locality (helps the 1D convolutions understand spatial relationships)
2. **Normalization** - Each parameter type gets normalized to zero mean and unit variance (computed from the training set)

The normalized data is what goes into the VAE. Details in `src/gaussvae/data/preprocessing.py`.

---

## Understanding the Checkpoint Format

If you're diving into the code, here's what's in those `.pt` files from Image-GS:

```python
import torch
checkpoint = torch.load('checkpoint.pt')

# The checkpoint contains these tensors:
checkpoint['xy']       # (N, 2) - XY positions
checkpoint['feat']     # (N, 3) - RGB colors  
checkpoint['scale']    # (N, 2) - Ellipse scales
checkpoint['rot']      # (N, 1) - Rotation angle
checkpoint['vis_feat'] # (N, 3) - Visualization colors (debug only, not used in rendering)
```

**Important note:** Only `xy`, `feat`, `scale`, and `rot` matter for rendering (8 params = 32 bytes per Gaussian). The `vis_feat` tensor is just random colors for debugging visualization and has zero effect on the actual rendered image - this was extensively verified during development.

---

## Choosing the Right Number of Gaussians

I ran benchmarks with different Gaussian counts to find the sweet spot. Here's what I found:

| Gaussians | Time/img | PSNR | SSIM | File Size | Notes |
|-----------|----------|------|------|-----------|-------|
| 100 | 33.2s | 28.62 | 0.7501 | 16 KB | Fast but lower quality |
| 200 | 30.2s | 27.66 | 0.8622 | 27 KB | Good balance for simple images |
| **512** | **28.9s** | **33.25** | **0.8859** | **60 KB** | **Chosen for this project** |
| 1000 | 34.4s | 34.42 | 0.9193 | 111 KB | Best quality, larger files |

**Interesting discovery:** Processing time stays around 30 seconds regardless of Gaussian count. This is because Image-GS automatically uses a minimum of 5000 optimization steps.

**Why I picked 512:**
- Power of 2 (nice for neural networks)
- Great quality-to-size ratio
- Processed all 11,501 images in about 4 days

---

## Docker Tips

When running Image-GS in Docker, you need to mount two volumes:

```bash
-v $(pwd)/data:/workspace/data                                    # Your input images
-v $(pwd)/third_party/image-gs:/workspace/third_party/image-gs    # Results and fonts
-w /workspace/third_party/image-gs                                # Set working directory
```

**Why both?** Image-GS writes results to `third_party/image-gs/results/` and needs the font files in `assets/fonts/` for rendering text overlays. Without both mounts, things will break in confusing ways!

---

## Project Organization

Here's how the codebase is organized (the important bits):

```
gaussvae/
├── data/delaunay/              # The dataset (11,501 abstract art images as Gaussians)
│   ├── npz/                    # Compressed NumPy format (173 MB)
│   │   ├── train/              # 10,502 samples for training
│   │   ├── dev/                # 500 samples for validation  
│   │   └── test/               # 499 samples for testing
│   └── normalization_stats.npz # Mean/std for each parameter type
│
├── src/gaussvae/               # Main Python package
│   ├── data/                   # Dataset loading and preprocessing
│   ├── models/                 # VAE architecture (encoder, decoder, layers)
│   └── training/               # Training loops and callbacks
│
├── scripts/                    # Utility scripts
│   ├── train_conv1d_vae.py     # Main training script
│   ├── encode_gaussians.py     # Batch encoding with Image-GS
│   └── convert_checkpoints_to_npz.py  # Convert .pt to .npz
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 08_vae_multi_image_demo.ipynb        # See VAE reconstructions
│   └── 02_architecture_prototyping.ipynb    # Original development
│
├── experiments/                # Training runs and analysis
│   ├── run_006_resnet_512d/    # Latest training experiment
│   └── loss_weighting_strategies.md  # Research notes
│
├── tests/                      # Test suite (173 tests passing)
│
├── configs/                    # Training configuration files
│
├── docker/                     # Dockerfiles for different environments
│   ├── Dockerfile.gaussvae-imagegs  # Image-GS rendering
│   └── Dockerfile.gaussvae-vae      # VAE training (TF + CUDA)
│
└── third_party/image-gs/       # Image-GS library (2D Gaussian Splatting)
    └── results/                # Generated Gaussian checkpoints
```

Want to understand how things work? Start with:
- `instructions/AGENT_HANDOFF.md` - Project status and next steps
- `instructions/VAE_ARCHITECTURE.md` - Technical architecture details  
- `data/delaunay/README.md` - Dataset documentation

---

## Testing

Want to make sure everything works? Run the test suite:

```bash
# Run all tests (requires GPU and gaussvae-imagegs Docker image)
pytest tests/ -v

# Just the GPU-dependent tests
pytest tests/test_imagegs_encode_decode.py -v
```

**What gets tested:**
- GPU encoding and decoding with Image-GS
- Checkpoint tensor structure validation
- VAE model architecture
- Data preprocessing pipeline

**Note for CI/CD:** Tests that need a GPU are marked with `@pytest.mark.gpu` and automatically skipped on GitHub Actions (since those runners don't have GPUs). The Docker builds still get validated though!

---

## Setting Up NVIDIA Stuff

If you're on a fresh machine and need to get NVIDIA drivers and Docker GPU support working:

**For Debian 11:**
```bash
echo "deb http://deb.debian.org/debian bullseye non-free" | sudo tee -a /etc/apt/sources.list.d/nvidia.list
sudo apt update
sudo apt install -y nvidia-driver firmware-misc-nonfree
sudo reboot
nvidia-smi  # Check it worked
```

**For Ubuntu:**
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi  # Check it worked
```

**Setting up nvidia-docker:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Quick test
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that test command shows your GPU info, you're good to go!

---

## Common Issues

**"command not found: docker-compose"**

You might need to install docker-compose separately:
```bash
sudo curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**Docker can't find the GPU**

Try restarting Docker:
```bash
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Need more help?**
- Image-GS issues: `third_party/image-gs/README.md`
- Docker setup: `docker/README.md`
- Dataset questions: `data/delaunay/README.md`

---

## Development Timeline

Here's how the project evolved:

**Phase 0: Figuring things out** (October 2025)
- Tested different Gaussian counts (100, 200, 500, 1000)
- Settled on 512 as the sweet spot
- Documented in `data/delaunay/BENCHMARK_RESULTS.md`

**Phase 1: Building the dataset** (October 2025)
- Processed 11,501 images to Gaussian splats
- Took about 4 days of GPU time on a Tesla T4
- 99.98% success rate (only 2 images failed)
- See `data/delaunay/DATASET_COMPLETE.md`

**Phase 2: Optimizing storage** (October 2025)
- Converted PyTorch checkpoints to compressed NumPy format
- Reduced dataset from 674 MB to 173 MB (lossless!)
- Verified bit-perfect conversion
- Details in `data/delaunay/CONVERSION_COMPLETE.md`

**Phase 3: VAE development** (October-November 2025)
- Built and tested the VAE architecture
- Ran 6 training experiments with different hyperparameters
- Discovered and diagnosed loss weighting issues
- Currently: Preparing training run #7 with fixes

**What's next:**
- Fix the loss weighting for rotation and color features
- Train for longer (previous runs were 25-40 epochs, want to push to 100+)
- Explore latent space interpolation and manipulation
- Maybe try some conditional generation?

---

## License and Acknowledgments

This project is released under the MIT License - see `LICENSE` for details.

**Third-party components:**
- Image-GS library (included in `third_party/`) - see `THIRD_PARTY_LICENSES.md`
- Various Python packages listed in `requirements.txt`

**Links:**
- Repository: https://github.com/rokocuba/GaussVae-showcase
- Report issues: https://github.com/rokocuba/GaussVae-showcase/issues

---

## Why This Project?

This started as an exploration of whether VAEs could learn meaningful compressed representations of Gaussian Splats. The intersection of generative models and novel rendering techniques is fascinating - can we learn to manipulate images in their Gaussian representation rather than pixel space?

It's also been a great excuse to:
- Get better at Docker and reproducible ML workflows
- Build a proper test suite for deep learning code
- Practice good documentation habits
- Learn about spatial data structures (Morton curves are cool!)

If you're interested in VAEs, Gaussian Splatting, or just want to see how to structure an ML research project, feel free to poke around the code and notebooks. Contributions and discussions are welcome!

---

**Last updated:** November 2025
