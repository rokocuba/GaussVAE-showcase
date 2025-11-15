# Docker Setup Guide

**Purpose:** Docker configurations for GaussVAE project

---

## ⚠️ Important: GPU Required

**Image-GS requires CUDA/GPU to run.** The Docker images build on any system (CPU-only), but:
- **fused-ssim** is NOT installed in the image (requires GPU to compile)
- **Image-GS execution** requires `--gpus all` flag
- **CI/CD** only validates builds, not execution

**To use Image-GS:**
1. Build image (works on any system)
2. Install fused-ssim at runtime with GPU
3. Run with `--gpus all`

See [Installation](#fused-ssim-installation) section below.

---

## Docker Images

| File | Purpose | Base | Size | GPU Required |
|------|---------|------|------|--------------|
| `Dockerfile.gaussvae-imagegs` | Image-GS processing | pytorch/pytorch:2.4.1-cuda12.4 | ~2GB | Yes (runtime) |
| `Dockerfile.gaussvae-vae` | VAE training + Jupyter | TensorFlow 2.12 + CUDA 11.8 | ~3 GB | Yes |

---

## Quick Start

### 1. Build Image

```bash
docker build -f docker/Dockerfile.gaussvae-imagegs -t gaussvae-imagegs .
```

**First build:** ~3-5 minutes (downloads base + installs dependencies)  
**Subsequent builds:** ~10 seconds (uses cached layers)

### 2. Install fused-ssim (GPU required)

**Option A: One-time install in running container**
```bash
docker run --gpus all -it gaussvae-imagegs bash
> git clone https://github.com/rahul-goel/fused-ssim.git /tmp/fused-ssim
> cd /tmp/fused-ssim && pip install .
> exit
# Then commit the container as new image (optional)
```

**Option B: Build extended image with fused-ssim**
```dockerfile
# Create Dockerfile.gaussvae-imagegs-gpu
FROM gaussvae-imagegs:latest
RUN git clone https://github.com/rahul-goel/fused-ssim.git /tmp/fused-ssim && \
    cd /tmp/fused-ssim && pip install . && \
    rm -rf /tmp/fused-ssim
```
```bash
docker build -f Dockerfile.gaussvae-imagegs-gpu -t gaussvae-imagegs:gpu .
```

**Why not in base image?**
- fused-ssim requires GPU to compile (uses CUDA `.cu` files)
- GitHub Actions CI/CD has no GPU → build would fail
- Base image validates builds, GPU image is for runtime

### 3. Run Image-GS

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/third_party/image-gs:/workspace/third_party/image-gs \
  -w /workspace/third_party/image-gs \
  gaussvae-imagegs \
  python main.py \
  --input_path /workspace/data/sample.png \
  --exp_name test/sample \
  --num_gaussians 100 \
  --max_steps 5000
```

---

## Architecture

### Two-Container Design

**gaussvae-imagegs (628 MB):**
- Base: Python 3.10-slim (122 MB)
- Packages: NumPy, Pillow, PyTorch, gsplat (~500 MB)
- Purpose: Fast Gaussian optimization

**gaussvae-vae (~3 GB):**
- Base: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
- Packages: TensorFlow 2.12.1, Jupyter Lab, NumPy, scikit-learn (~2.3 GB)
- Purpose: VAE training + interactive development
- Status: ✅ Operational (prototype complete, 2025-11-01)

**Why separate?**
- Independent updates (change VAE without rebuilding Image-GS)
- Size optimization (Image-GS stays lightweight)
- Clear separation (image processing vs ML training)

---

## Volume Mounting

**Critical for Image-GS:**
```bash
-v $(pwd)/data:/workspace/data                                 # Input images
-v $(pwd)/third_party/image-gs:/workspace/third_party/image-gs # Results + assets
-w /workspace/third_party/image-gs                             # Working dir (needs fonts)
```

**Why both volumes?**
- Image-GS writes results to `third_party/image-gs/results/`
- Requires `assets/fonts/` for rendering
- Without `-v third_party/image-gs`, results lost when container exits

**How it works:**
```
Host: data/sample.png
  ↓ (mount)
Container: /workspace/data/sample.png
  ↓ (Image-GS optimization)
Container: /workspace/third_party/image-gs/results/.../checkpoint.pt
  ↓ (mount)
Host: third_party/image-gs/results/.../checkpoint.pt
```

Same folder, not a copy. Changes appear instantly on host.

---

## Docker Layer Caching

| Build Status | What Happens | Time |
|-------------|--------------|------|
| **First build** | Downloads base + installs all | 3-5 min |
| **No changes** | Uses cached image | 1-2 sec |
| **Code changed** | Rebuilds only code layer | ~10 sec |
| **Dependencies changed** | Rebuilds from that layer | ~2 min |

**Tip:** Don't use `--no-cache` during development - wastes time.

---

## Troubleshooting

### "docker-compose: command not found"

```bash
# Linux
sudo curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

### SSL Certificate Errors

**Option 1: Use system packages (recommended)**
```dockerfile
RUN apt-get update && apt-get install -y python3-numpy python3-pillow
```

**Option 2: Disable SSL (development only)**
```dockerfile
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy pillow
```

**Option 3: Use conda**
```dockerfile
FROM continuumio/miniconda3
RUN conda install numpy pillow -y
```

### Permission Denied

```bash
# Give ownership to your user
sudo chown -R $USER:$USER data/

# Or run with your UID
docker run --user $(id -u):$(id -g) --rm ...
```

### Out of Disk Space

```bash
# Check usage
docker system df

# Clean up
docker container prune  # Remove stopped containers
docker image prune -a   # Remove unused images
docker system prune -a  # Remove everything unused (careful!)
```

### GPU Not Accessible

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# If fails, check nvidia-docker2
sudo apt install nvidia-docker2
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Development Workflow

### Interactive Container

```bash
# Explore container interactively
docker run -it --rm gaussvae-imagegs /bin/bash

# Inside container:
ls -la
python --version
pip list
```

### View Logs

```bash
# Running container
docker logs <container_id>

# docker-compose
docker-compose logs imagegs
```

### Custom Commands

```bash
# Run different command
docker run --rm gaussvae-imagegs python -c "import torch; print(torch.__version__)"
```

---

## When to Use Docker

**Use Docker for:**
- Reproducible environments
- CI/CD pipelines
- Sharing with collaborators
- Deployment

**Skip Docker for:**
- Active debugging (native Python easier)
- Jupyter notebooks (volume mount complications)
- When GPU overhead matters (minimal, but exists)

---

## CI/CD Integration

GitHub Actions installs docker-compose (not included by default):

```yaml
- name: Install Docker Compose
  run: |
    sudo curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
      -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
```

---

## Performance

| Method | First Run | Subsequent | Pros | Cons |
|--------|-----------|------------|------|------|
| Docker | 3-5 min | 1-2 sec | Reproducible | Slight overhead |
| Native Python | Instant | Instant | Fastest | Dependency conflicts |

---

## References

- Main README: `../README.md`
- Image-GS Details: `../instructions/IMAGE_GS_EXPLAINED.md`
- Operations Guide: `../instructions/OPERATIONS.md`
- Testing: `../tests/README.md`
