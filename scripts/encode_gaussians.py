#!/usr/bin/env python3
"""
General-purpose Gaussian Splatting encoder using Image-GS.

Encodes PNG images to Gaussian Splat checkpoints (.pt files).
Supports single image or batch (directory) processing with optional rendering.

Features:
- Single image or batch directory processing
- Optional rendering: Saves original + reconstructed images for comparison
- Resumability: Skip already-processed images
- Progress tracking: ETA, percentage, success/failure counts
- Error handling: Continue on failure, detailed logs
- Flexible output: Organized directory structure
- JSONL logging: Append-only processing logs

Usage:
    # Single image (encode only)
    python scripts/encode_gaussians.py \\
        --input path/to/image.png \\
        --output path/to/output/dir \\
        --num_gaussians 512

    # Batch directory (encode only)
    python scripts/encode_gaussians.py \\
        --input path/to/images/ \\
        --output path/to/output/ \\
        --num_gaussians 512

    # With rendering (saves original + rendered images)
    python scripts/encode_gaussians.py \\
        --input path/to/images/ \\
        --output path/to/output/ \\
        --num_gaussians 512 \\
        --eval

    # With quantization (half-precision, smaller files)
    python scripts/encode_gaussians.py \\
        --input path/to/images/ \\
        --output path/to/output/ \\
        --num_gaussians 512 \\
        --quantize

Output structure:
    output/
    ├── checkpoints/           # .pt checkpoint files
    │   ├── img_000001.pt
    │   └── img_000002.pt
    ├── images/                # Only if --eval is used
    │   ├── original/          # Original input images
    │   │   ├── img_000001.png
    │   │   └── img_000002.png
    │   └── rendered/          # Reconstructed images from checkpoints
    │       ├── img_000001.png
    │       └── img_000002.png
    ├── processing_log.jsonl   # Per-image processing details
    └── summary.json           # Overall statistics
"""

import argparse
import json
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
import sys
import re


def parse_log_file(log_path):
    """Parse Image-GS log file to extract final metrics."""
    metrics = {}
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            # Look for final step metrics
            for line in reversed(lines):
                if "PSNR:" in line and "SSIM:" in line and "LPIPS:" in line:
                    psnr_match = re.search(r"PSNR:\s*([\d.]+)", line)
                    ssim_match = re.search(r"SSIM:\s*([\d.]+)", line)
                    lpips_match = re.search(r"LPIPS:\s*([\d.]+)", line)

                    if psnr_match:
                        metrics["psnr"] = float(psnr_match.group(1))
                    if ssim_match:
                        metrics["ssim"] = float(ssim_match.group(1))
                    if lpips_match:
                        metrics["lpips"] = float(lpips_match.group(1))

                    if metrics:
                        break
    except Exception:
        pass  # Silent fail, metrics will be empty
    return metrics


def get_file_size(path):
    """Get file size in bytes."""
    try:
        return path.stat().st_size
    except:
        return None


def load_processed_images(log_path):
    """Load set of already-processed image IDs from JSONL log."""
    processed = set()
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("success"):
                            processed.add(entry["image_id"])
        except Exception as e:
            print(f"Warning: Could not read log file: {e}")
    return processed


def append_log_entry(log_path, entry):
    """Append single entry to JSONL log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_summary(summary_path, summary_data):
    """Save summary JSON."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_eta(processed_count, total_count, elapsed_time):
    """Estimate time remaining."""
    if processed_count == 0:
        return "Calculating..."

    avg_time_per_image = elapsed_time / processed_count
    remaining = total_count - processed_count
    eta_seconds = remaining * avg_time_per_image

    return format_time(eta_seconds)


def encode_image(
    image_path,
    output_base_dir,
    num_gaussians,
    max_steps,
    quantize,
    eval_mode,
    workspace_root,
):
    """
    Encode a single image using Image-GS Docker container.

    Args:
        image_path: Path to input PNG image
        output_base_dir: Base output directory (will create checkpoints/, images/ subdirs)
        num_gaussians: Number of Gaussians
        max_steps: Maximum optimization steps
        quantize: Enable half-precision quantization (16-bit)
        eval_mode: If True, render reconstructed image and save original + rendered
        workspace_root: Root directory for Docker volume mounts

    Returns dict with:
        - success: bool
        - duration: float (seconds)
        - metrics: dict (PSNR, SSIM, LPIPS)
        - checkpoint_size: int (bytes)
        - has_render: bool (True if render was saved)
        - error: str (if failed)
    """
    image_id = image_path.stem
    exp_name = f"encode_gaussians/{image_id}"

    # Setup output paths
    checkpoint_dir = output_base_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"{image_id}.pt"

    if eval_mode:
        original_dir = output_base_dir / "images" / "original"
        rendered_dir = output_base_dir / "images" / "rendered"
        original_path = original_dir / f"{image_id}.png"
        rendered_path = rendered_dir / f"{image_id}.png"
    else:
        original_path = None
        rendered_path = None

    # Construct relative path from workspace root
    try:
        rel_input_path = image_path.relative_to(workspace_root)
    except ValueError:
        # If image is not under workspace root, use absolute path
        rel_input_path = image_path.absolute()

    # Docker command - mount workspace root as read-only for input access,
    # and third_party/image-gs separately for write access to results
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{workspace_root}:/workspace_ro:ro",  # Read-only mount for input files
        "-v",
        f"{workspace_root}/third_party/image-gs:/workspace/third_party/image-gs",  # RW mount for results
        "-w",
        "/workspace/third_party/image-gs",
        "gaussvae-imagegs",
        "python",
        "main.py",
        f"--input_path=/workspace_ro/{rel_input_path}",
        f"--exp_name={exp_name}",
        f"--num_gaussians={num_gaussians}",
        f"--max_steps={max_steps}",
    ]

    # Add quantization if requested
    if quantize:
        cmd.append("--quantize")

    # Note: We don't pass --eval to Image-GS because it expects an existing checkpoint
    # Image-GS always generates rendered images during training, so we'll copy them
    # if eval_mode is True

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600  # 10 minute timeout
        )
        duration = time.time() - start_time

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return {
                "success": False,
                "duration": duration,
                "error": error_msg,
            }

        # Find result files in Image-GS output directory
        result_base = workspace_root / "third_party/image-gs/results" / exp_name
        matching_dirs = list(
            result_base.parent.glob(f"{result_base.name}/num-{num_gaussians}_*")
        )

        if not matching_dirs:
            return {
                "success": False,
                "duration": duration,
                "error": f"Result directory not found: {result_base}",
            }

        actual_result_dir = matching_dirs[0]

        # Find checkpoint
        checkpoint_dir = actual_result_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("ckpt_step-*.pt"))

        if not checkpoints:
            return {
                "success": False,
                "duration": duration,
                "error": "No checkpoint found",
            }

        source_checkpoint = checkpoints[0]
        checkpoint_size = get_file_size(source_checkpoint)

        # Parse metrics from log
        log_path = actual_result_dir / "log_train.txt"
        metrics = parse_log_file(log_path)

        # Copy checkpoint to final location
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, checkpoint_path)

        # Handle rendering if eval_mode is enabled
        has_render = False
        if eval_mode and original_path and rendered_path:
            # Find rendered image (Image-GS saves as render_res-WxH.png with variable resolution)
            rendered_files = list(actual_result_dir.glob("render_res-*.png"))
            if rendered_files:
                source_render = rendered_files[0]  # Take first match

                # Copy original and rendered images
                original_path.parent.mkdir(parents=True, exist_ok=True)
                rendered_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(image_path, original_path)
                shutil.copy2(source_render, rendered_path)
                has_render = True

        return {
            "success": True,
            "duration": duration,
            "metrics": metrics,
            "checkpoint_size": checkpoint_size,
            "has_render": has_render,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "error": "Timeout (>10 minutes)",
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Encode PNG images to Gaussian Splat checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image (encode only)
  python scripts/encode_gaussians.py \\
      --input data/image.png \\
      --output data/output/ \\
      --num_gaussians 512
  
  # Batch directory (encode only)
  python scripts/encode_gaussians.py \\
      --input data/images/ \\
      --output data/output/ \\
      --num_gaussians 512
  
  # With rendering (saves original + rendered for comparison)
  python scripts/encode_gaussians.py \\
      --input data/images/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --eval
  
  # With quantization (16-bit, smaller files)
  python scripts/encode_gaussians.py \\
      --input data/images/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --quantize
  
  # Full options
  python scripts/encode_gaussians.py \\
      --input data/images/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --max_steps 5000 \\
      --quantize \\
      --eval
  
  # Dry run (preview what will be processed)
  python scripts/encode_gaussians.py \\
      --input data/images/ \\
      --output data/output/ \\
      --dry_run

Output structure:
  output/
  ├── checkpoints/          # .pt checkpoint files
  ├── images/               # Only if --eval is used
  │   ├── original/         # Original input images
  │   └── rendered/         # Reconstructed images
  ├── processing_log.jsonl  # Per-image details
  └── summary.json          # Overall statistics
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input PNG image or directory containing PNG images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_gaussians",
        type=int,
        default=512,
        help="Number of Gaussians (default: 512)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Optimization steps (default: 5000, Image-GS minimum)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable half-precision quantization (16-bit, reduces file size ~50%%)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Render reconstructed images and save original + rendered for comparison",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL log file path (default: {output}/processing_log.jsonl)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess already-completed images (overwrite checkpoints)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only errors and final summary)",
    )

    args = parser.parse_args()

    # Setup paths
    workspace_root = Path.cwd()
    input_path = args.input.absolute()
    output_dir = args.output.absolute()

    # Determine if single image or batch
    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            print(f"Error: Input file must be PNG, got: {input_path.suffix}")
            sys.exit(1)
        images = [input_path]
        mode = "single"
    elif input_path.is_dir():
        images = sorted(input_path.glob("*.png"))
        mode = "batch"
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if not images:
        print(f"Error: No PNG images found at: {input_path}")
        sys.exit(1)

    # Setup log path
    if args.log is None:
        log_path = output_dir / "processing_log.jsonl"
    else:
        log_path = args.log.absolute()

    summary_path = output_dir / "summary.json"

    # Load already-processed images (for resumability)
    processed_images = set() if args.force else load_processed_images(log_path)

    # Filter out already-processed images
    images_to_process = [img for img in images if img.stem not in processed_images]

    # Print configuration
    if not args.quiet:
        print("=" * 70)
        print(f"Gaussian Splatting Encoder - {mode.upper()} Mode")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Gaussians: {args.num_gaussians}")
        print(f"  Max steps: {args.max_steps}")
        print(
            f"  Quantization: {'Enabled (16-bit)' if args.quantize else 'Disabled (32-bit)'}"
        )
        print(
            f"  Render mode: {'Enabled (saves original + rendered)' if args.eval else 'Disabled'}"
        )
        print(f"  Input: {input_path}")
        print(f"  Output: {output_dir}")
        print(f"  Log: {log_path}")
        print()
        print(f"Images:")
        print(f"  Total: {len(images)}")
        print(f"  Already processed: {len(processed_images)}")
        print(f"  To process: {len(images_to_process)}")
        print()
        print(f"Output structure:")
        print(f"  {output_dir}/checkpoints/          # .pt files")
        if args.eval:
            print(f"  {output_dir}/images/original/      # Original PNGs")
            print(f"  {output_dir}/images/rendered/      # Reconstructed PNGs")
        print("=" * 70)

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No images will be processed]")
        if images_to_process:
            print(f"\nWould process {len(images_to_process)} images:")
            for i, img in enumerate(images_to_process[:10], 1):
                print(f"  {i}. {img.name}")
            if len(images_to_process) > 10:
                print(f"  ... and {len(images_to_process) - 10} more")
        else:
            print("\nNo images to process (all already completed).")
        sys.exit(0)

    # Check if nothing to process
    if not images_to_process:
        if not args.quiet:
            print("\nAll images already processed!")
            print(f"  Use --force to reprocess all {len(images)} images.")
        sys.exit(0)

    # Confirm before starting (batch mode only)
    if mode == "batch" and not args.quiet:
        print(f"\nAbout to process {len(images_to_process)} images...")
        estimated_time = len(images_to_process) * 31  # ~31s per image
        print(
            f"Estimated time: {format_time(estimated_time)} ({estimated_time/3600:.1f} hours)"
        )
        print()
        response = input("Continue? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Process images
    if not args.quiet:
        print("\n" + "=" * 70)
        print("PROCESSING")
        print("=" * 70)

    start_time = time.time()
    success_count = 0
    failure_count = 0

    for i, image_path in enumerate(images_to_process, 1):
        image_id = image_path.stem

        # Progress header (skip in quiet mode or single image)
        if not args.quiet and mode == "batch":
            elapsed = time.time() - start_time
            progress_pct = (i - 1) / len(images_to_process) * 100
            eta = estimate_eta(i - 1, len(images_to_process), elapsed)

            print(
                f"\n[{i}/{len(images_to_process)}] {progress_pct:.1f}% | ETA: {eta} | {image_path.name}"
            )
            print(f"  Success: {success_count} | Failed: {failure_count}")
        elif not args.quiet:
            print(f"Processing {image_path.name}...")

        # Encode image
        result = encode_image(
            image_path,
            output_dir,
            args.num_gaussians,
            args.max_steps,
            args.quantize,
            args.eval,
            workspace_root,
        )

        # Update counts
        if result["success"]:
            success_count += 1
            if not args.quiet:
                psnr = result["metrics"].get("psnr", 0)
                ssim = result["metrics"].get("ssim", 0)
                ckpt_size = (
                    result["checkpoint_size"] / 1024 if result["checkpoint_size"] else 0
                )
                status_line = f"  ✓ {result['duration']:.1f}s | PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | {ckpt_size:.1f} KB"
                if args.eval:
                    status_line += (
                        f" | Render: {'✓' if result.get('has_render') else '✗'}"
                    )
                print(status_line)
        else:
            failure_count += 1
            if not args.quiet:
                print(
                    f"  ✗ FAILED ({result['duration']:.1f}s): {result.get('error', 'Unknown')[:100]}"
                )

        # Log entry
        log_entry = {
            "image_id": image_id,
            "image_name": image_path.name,
            "num_gaussians": args.num_gaussians,
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        append_log_entry(log_path, log_entry)

        # Save summary periodically (every 10 images or at end)
        if mode == "batch" and (i % 10 == 0 or i == len(images_to_process)):
            total_processed = len(processed_images) + i
            summary = {
                "num_gaussians": args.num_gaussians,
                "max_steps": args.max_steps,
                "total_images": len(images),
                "processed": total_processed,
                "remaining": len(images) - total_processed,
                "success_count": success_count,
                "failure_count": failure_count,
                "last_updated": datetime.now().isoformat(),
            }
            save_summary(summary_path, summary)

    # Final summary
    total_time = time.time() - start_time

    if not args.quiet:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

    print(f"Total images: {len(images)}")
    print(f"Processed this run: {len(images_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total time: {format_time(total_time)}")

    if success_count > 0:
        avg_time = total_time / success_count
        print(f"Average time per image: {avg_time:.1f}s")

    if not args.quiet:
        print(f"\nCheckpoints: {output_dir}/checkpoints/")
        if args.eval:
            print(f"Original images: {output_dir}/images/original/")
            print(f"Rendered images: {output_dir}/images/rendered/")
        print(f"Log: {log_path}")
        if mode == "batch":
            print(f"Summary: {summary_path}")
        print("=" * 70)

    if failure_count > 0:
        print(f"\n⚠ {failure_count} images failed. Check log for details.")
        sys.exit(1)
    else:
        if not args.quiet:
            print("\n✓ All images processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
