#!/usr/bin/env python3
"""
General-purpose Gaussian Splatting decoder using Image-GS.

Renders Gaussian Splat checkpoints (.pt files) to PNG images.
Supports single checkpoint or batch (directory) processing with flexible rendering options.

Features:
- Single checkpoint or batch directory processing
- Custom resolution rendering (upscaling/downscaling)
- Resumability: Skip already-rendered checkpoints
- Progress tracking: ETA, percentage, success/failure counts
- Error handling: Continue on failure, detailed logs
- Flexible output: Organized directory structure
- JSONL logging: Append-only processing logs

Usage:
    # Single checkpoint (original resolution)
    python scripts/decode_gaussians.py \\
        --input path/to/checkpoint.pt \\
        --output path/to/output/dir

    # Batch directory (original resolution)
    python scripts/decode_gaussians.py \\
        --input path/to/checkpoints/ \\
        --output path/to/output/

    # Custom resolution (height=2048, maintains aspect ratio)
    python scripts/decode_gaussians.py \\
        --input path/to/checkpoints/ \\
        --output path/to/output/ \\
        --render_height 2048

    # With quantization settings (must match checkpoint)
    python scripts/decode_gaussians.py \\
        --input path/to/checkpoints/ \\
        --output path/to/output/ \\
        --quantize

Output structure:
    output/
    ├── images/                # Rendered PNG images
    │   ├── img_000001.png
    │   └── img_000002.png
    ├── processing_log.jsonl   # Per-checkpoint processing details
    └── summary.json           # Overall statistics
"""

import argparse
import json
import subprocess
import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import sys
import re


def parse_log_file(log_path):
    """Parse Image-GS log file to extract rendering metrics."""
    metrics = {}
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            # Look for rendering completion message
            for line in reversed(lines):
                if "Rendering at resolution" in line:
                    # Extract resolution
                    res_match = re.search(r"\((\d+), (\d+)\)", line)
                    if res_match:
                        metrics["render_height"] = int(res_match.group(1))
                        metrics["render_width"] = int(res_match.group(2))
                    break
                if "Time:" in line:
                    time_match = re.search(r"Time:\s*([\d.]+)\s*s", line)
                    if time_match:
                        metrics["render_time"] = float(time_match.group(1))
    except Exception:
        pass  # Silent fail, metrics will be empty
    return metrics


def get_file_size(path):
    """Get file size in bytes."""
    try:
        return path.stat().st_size
    except:
        return None


def load_processed_checkpoints(log_path):
    """Load set of already-processed checkpoint IDs from JSONL log."""
    processed = set()
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("success"):
                            processed.add(entry["checkpoint_id"])
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

    avg_time_per_checkpoint = elapsed_time / processed_count
    remaining = total_count - processed_count
    eta_seconds = remaining * avg_time_per_checkpoint

    return format_time(eta_seconds)


def decode_checkpoint(
    checkpoint_path,
    output_base_dir,
    num_gaussians,
    quantize,
    render_height,
    workspace_root,
    source_image_path=None,
):
    """
    Decode a single checkpoint using Image-GS Docker container.

    Args:
        checkpoint_path: Path to input .pt checkpoint file
        output_base_dir: Base output directory (will create images/ subdir)
        num_gaussians: Number of Gaussians (must match checkpoint)
        quantize: Enable half-precision quantization (must match checkpoint)
        render_height: Custom rendering height (None = original resolution)
        workspace_root: Root directory for Docker volume mounts
        source_image_path: Path to source image file (required by Image-GS for eval mode)

    Returns dict with:
        - success: bool
        - duration: float (seconds)
        - metrics: dict (render_time, resolution)
        - image_size: int (bytes)
        - error: str (if failed)
    """
    checkpoint_id = checkpoint_path.stem
    exp_name = f"decode_gaussians/{checkpoint_id}"

    # Setup output paths
    output_dir = output_base_dir / "images"
    output_image_path = output_dir / f"{checkpoint_id}.png"

    # Create temporary directory structure for Image-GS results
    # Image-GS expects: results/{exp_name}/{config}/checkpoints/ckpt_step-*.pt
    # Note: Docker creates root-owned files, so we'll manually clean up with sudo rm
    temp_dir = tempfile.mkdtemp(prefix="imagegs_decode_")
    try:
        temp_path = Path(temp_dir)
        
        # Create Image-GS directory structure in results path
        # Generate config string to match Image-GS naming convention
        config = f"num-{num_gaussians}"
        if quantize:
            config += "_bits-16-16-16-16"
        else:
            config += "_bits-32-32-32-32"
        config += "_inv-scale-5.0_top-10_g-0.3_l1-1.0_l2-0.0_ssim-0.1_decay-1-10.0_prog"  # Default Image-GS config
        
        # Create structure: results/{exp_name}/{config}/checkpoints/
        results_base = temp_path / "results"
        result_dir = results_base / exp_name / config
        checkpoint_dir = result_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint to expected location with expected name
        temp_checkpoint = checkpoint_dir / "ckpt_step-0.pt"
        shutil.copy2(checkpoint_path, temp_checkpoint)

        # Setup source image if provided
        if source_image_path is None:
            return {
                "success": False,
                "duration": 0.0,
                "error": "Source image path required for Image-GS rendering",
            }

        # Copy source image to temp data directory (outside Image-GS workspace)
        temp_data_dir = temp_path / "data"
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        temp_source_image = temp_data_dir / source_image_path.name
        shutil.copy2(source_image_path, temp_source_image)

        # Docker command - mount Image-GS code and separate data/results directories
        imagegs_path = workspace_root / "third_party" / "image-gs"
        
        # Checkpoint path inside container
        container_ckpt_path = f"/workspace/imagegs_results/{exp_name}/{config}/checkpoints/ckpt_step-0.pt"
        
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{imagegs_path}:/workspace/third_party/image-gs:ro",  # Image-GS code (read-only)
            "-v",
            f"{temp_path}/results:/workspace/imagegs_results",  # Results dir (read-write, outside image-gs)
            "-v",
            f"{temp_path}/data:/workspace/imagegs_data",  # Data dir with source image (read-write, outside image-gs)
            "-w",
            "/workspace/third_party/image-gs",
            "gaussvae-imagegs",
            "python",
            "main.py",
            f"--exp_name={exp_name}",
            f"--num_gaussians={num_gaussians}",
            f"--input_path={source_image_path.name}",  # Relative to data_root
            f"--data_root=/workspace/imagegs_data",  # Mounted temp data directory
            f"--log_root=/workspace/imagegs_results",  # Mounted temp results directory
            f"--ckpt_file={container_ckpt_path}",  # Explicit checkpoint path for eval mode
            "--eval",  # Render mode
        ]

        # Add quantization if requested
        if quantize:
            cmd.append("--quantize")

        # Add custom rendering height if specified
        if render_height is not None:
            cmd.append(f"--render_height={render_height}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120  # 2 minute timeout
            )
            duration = time.time() - start_time

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                return {
                    "success": False,
                    "duration": duration,
                    "error": error_msg,
                }

            # Find rendered image in eval directory
            # Image-GS saves renders to: {log_root}/{exp_name}/{config}/eval/render_*.png
            # Use dynamic discovery like encoder script (config string varies)
            result_base = results_base / exp_name
            
            matching_dirs = list(result_base.glob(f"num-{num_gaussians}_*"))
            
            if not matching_dirs:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Result directory not found. Looked in: {result_base}, exists: {result_base.exists()}",
                }
            
            # Find the directory that has an eval subdirectory (Image-GS may create multiple config dirs)
            actual_result_dir = None
            for d in matching_dirs:
                eval_test = d / "eval"
                if eval_test.exists():
                    actual_result_dir = d
                    break
            
            if actual_result_dir is None:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"No result directory with eval subdirectory found. Searched: {matching_dirs}",
                }
            
            eval_dir = actual_result_dir / "eval"
            
            rendered_files = list(eval_dir.glob("render_*.png"))

            if not rendered_files:
                return {
                    "success": False,
                    "duration": duration,
                    "error": "No rendered image found in eval directory",
                }

            source_image = rendered_files[0]
            image_size = get_file_size(source_image)

            # Parse metrics from log (in eval mode, log is log_eval.txt)
            log_path = actual_result_dir / "log_eval.txt"
            metrics = parse_log_file(log_path)

            # Copy rendered image to final location
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, output_image_path)

            return {
                "success": True,
                "duration": duration,
                "metrics": metrics,
                "image_size": image_size,
            }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "duration": duration,
                "error": "Timeout (>2 minutes)",
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "duration": duration,
                "error": str(e),
            }
    finally:
        # Clean up temp directory (Docker may create root-owned files)
        # Use subprocess to ensure we can delete root-owned files
        try:
            subprocess.run(['rm', '-rf', temp_dir], check=False, capture_output=True)
        except Exception:
            pass  # Best effort cleanup


def main():
    parser = argparse.ArgumentParser(
        description="Decode Gaussian Splat checkpoints to PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single checkpoint (original resolution)
  python scripts/decode_gaussians.py \\
      --input data/checkpoint.pt \\
      --output data/output/ \\
      --num_gaussians 512
  
  # Batch directory (original resolution)
  python scripts/decode_gaussians.py \\
      --input data/checkpoints/ \\
      --output data/output/ \\
      --num_gaussians 512
  
  # Custom resolution (2048px height, maintains aspect ratio)
  python scripts/decode_gaussians.py \\
      --input data/checkpoints/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --render_height 2048
  
  # With quantization (16-bit, must match checkpoint)
  python scripts/decode_gaussians.py \\
      --input data/checkpoints/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --quantize
  
  # Full options
  python scripts/decode_gaussians.py \\
      --input data/checkpoints/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --quantize \\
      --render_height 2048
  
  # Dry run (preview what will be processed)
  python scripts/decode_gaussians.py \\
      --input data/checkpoints/ \\
      --output data/output/ \\
      --num_gaussians 512 \\
      --dry_run

Output structure:
  output/
  ├── images/               # Rendered PNG images
  ├── processing_log.jsonl  # Per-checkpoint details
  └── summary.json          # Overall statistics
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .pt checkpoint or directory containing .pt checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--input_images",
        type=Path,
        required=True,
        help="Directory containing source PNG images (checkpoint IDs must match image filenames)",
    )
    parser.add_argument(
        "--num_gaussians",
        type=int,
        default=512,
        help="Number of Gaussians (must match checkpoint, default: 512)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable half-precision quantization (must match checkpoint settings)",
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=None,
        help="Custom rendering height in pixels (maintains aspect ratio, default: original)",
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
        help="Reprocess already-completed checkpoints (overwrite images)",
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
    input_images_dir = args.input_images.absolute()

    # Verify input_images directory exists
    if not input_images_dir.exists() or not input_images_dir.is_dir():
        print(f"Error: Input images directory does not exist: {input_images_dir}")
        sys.exit(1)

    # Determine if single checkpoint or batch
    if input_path.is_file():
        if input_path.suffix.lower() != ".pt":
            print(f"Error: Input file must be .pt checkpoint, got: {input_path.suffix}")
            sys.exit(1)
        checkpoints = [input_path]
        mode = "single"
    elif input_path.is_dir():
        checkpoints = sorted(input_path.glob("*.pt"))
        mode = "batch"
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if not checkpoints:
        print(f"Error: No .pt checkpoints found at: {input_path}")
        sys.exit(1)

    # Build checkpoint ID -> source image mapping
    checkpoint_to_image = {}
    for ckpt in checkpoints:
        # For checkpoints in subdirectories, use parent directory name as image ID
        # E.g., data/results/100_gaussians/1023/checkpoint.pt -> image ID = 1023
        if ckpt.name == "checkpoint.pt" and ckpt.parent.name.isdigit():
            image_id = ckpt.parent.name
        else:
            # For standalone checkpoints, use stem
            image_id = ckpt.stem
        
        # Try to find matching image: image_id.png
        source_image = input_images_dir / f"{image_id}.png"
        if source_image.exists():
            checkpoint_to_image[ckpt.stem] = source_image
        else:
            print(f"Warning: No source image found for checkpoint {ckpt.name} (image ID: {image_id}) at {source_image}")

    if not checkpoint_to_image:
        print(f"Error: No matching source images found in {input_images_dir}")
        print(f"Expected image filenames to match checkpoint IDs (e.g., checkpoint.pt -> checkpoint.png)")
        sys.exit(1)

    # Setup log path
    if args.log is None:
        log_path = output_dir / "processing_log.jsonl"
    else:
        log_path = args.log.absolute()

    summary_path = output_dir / "summary.json"

    # Load already-processed checkpoints (for resumability)
    processed_checkpoints = set() if args.force else load_processed_checkpoints(log_path)

    # Filter out already-processed checkpoints (and ones without source images)
    checkpoints_to_process = [
        ckpt for ckpt in checkpoints 
        if ckpt.stem not in processed_checkpoints and ckpt.stem in checkpoint_to_image
    ]

    # Print configuration
    if not args.quiet:
        print("=" * 70)
        print(f"Gaussian Splatting Decoder - {mode.upper()} Mode")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Gaussians: {args.num_gaussians}")
        print(
            f"  Quantization: {'Enabled (16-bit)' if args.quantize else 'Disabled (32-bit)'}"
        )
        render_info = (
            f"Custom ({args.render_height}px height)"
            if args.render_height
            else "Original resolution"
        )
        print(f"  Render mode: {render_info}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_dir}")
        print(f"  Log: {log_path}")
        print()
        print(f"Checkpoints:")
        print(f"  Total: {len(checkpoints)}")
        print(f"  Already processed: {len(processed_checkpoints)}")
        print(f"  To process: {len(checkpoints_to_process)}")
        print()
        print(f"Output structure:")
        print(f"  {output_dir}/images/          # Rendered PNG images")
        print("=" * 70)

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No checkpoints will be processed]")
        if checkpoints_to_process:
            print(f"\nWould process {len(checkpoints_to_process)} checkpoints:")
            for i, ckpt in enumerate(checkpoints_to_process[:10], 1):
                print(f"  {i}. {ckpt.name}")
            if len(checkpoints_to_process) > 10:
                print(f"  ... and {len(checkpoints_to_process) - 10} more")
        else:
            print("\nNo checkpoints to process (all already completed).")
        sys.exit(0)

    # Check if nothing to process
    if not checkpoints_to_process:
        if not args.quiet:
            print("\nAll checkpoints already processed!")
            print(f"  Use --force to reprocess all {len(checkpoints)} checkpoints.")
        sys.exit(0)

    # Confirm before starting (batch mode only)
    if mode == "batch" and not args.quiet:
        print(f"\nAbout to process {len(checkpoints_to_process)} checkpoints...")
        estimated_time = len(checkpoints_to_process) * 5  # ~5s per checkpoint
        print(
            f"Estimated time: {format_time(estimated_time)} ({estimated_time/60:.1f} minutes)"
        )
        print()
        response = input("Continue? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Process checkpoints
    if not args.quiet:
        print("\n" + "=" * 70)
        print("PROCESSING")
        print("=" * 70)

    start_time = time.time()
    success_count = 0
    failure_count = 0

    for i, checkpoint_path in enumerate(checkpoints_to_process, 1):
        checkpoint_id = checkpoint_path.stem

        # Progress header (skip in quiet mode or single checkpoint)
        if not args.quiet and mode == "batch":
            elapsed = time.time() - start_time
            progress_pct = (i - 1) / len(checkpoints_to_process) * 100
            eta = estimate_eta(i - 1, len(checkpoints_to_process), elapsed)

            print(
                f"\n[{i}/{len(checkpoints_to_process)}] {progress_pct:.1f}% | ETA: {eta} | {checkpoint_path.name}"
            )
            print(f"  Success: {success_count} | Failed: {failure_count}")
        elif not args.quiet:
            print(f"Processing {checkpoint_path.name}...")

        # Get source image path for this checkpoint
        source_image = checkpoint_to_image.get(checkpoint_id)

        # Decode checkpoint
        result = decode_checkpoint(
            checkpoint_path,
            output_dir,
            args.num_gaussians,
            args.quantize,
            args.render_height,
            workspace_root,
            source_image_path=source_image,
        )

        # Update counts
        if result["success"]:
            success_count += 1
            if not args.quiet:
                render_time = result["metrics"].get("render_time", 0)
                img_size = (
                    result["image_size"] / 1024 if result["image_size"] else 0
                )
                render_h = result["metrics"].get("render_height", "?")
                render_w = result["metrics"].get("render_width", "?")
                status_line = f"  ✓ {result['duration']:.1f}s | Render: {render_time:.3f}s | Resolution: {render_h}x{render_w} | {img_size:.1f} KB"
                print(status_line)
        else:
            failure_count += 1
            if not args.quiet:
                print(
                    f"  ✗ FAILED ({result['duration']:.1f}s): {result.get('error', 'Unknown')[:100]}"
                )

        # Log entry
        log_entry = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_name": checkpoint_path.name,
            "num_gaussians": args.num_gaussians,
            "render_height": args.render_height,
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        append_log_entry(log_path, log_entry)

        # Save summary periodically (every 10 checkpoints or at end)
        if mode == "batch" and (i % 10 == 0 or i == len(checkpoints_to_process)):
            total_processed = len(processed_checkpoints) + i
            summary = {
                "num_gaussians": args.num_gaussians,
                "render_height": args.render_height,
                "total_checkpoints": len(checkpoints),
                "processed": total_processed,
                "remaining": len(checkpoints) - total_processed,
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

    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Processed this run: {len(checkpoints_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total time: {format_time(total_time)}")

    if success_count > 0:
        avg_time = total_time / success_count
        print(f"Average time per checkpoint: {avg_time:.1f}s")

    if not args.quiet:
        print(f"\nRendered images: {output_dir}/images/")
        print(f"Log: {log_path}")
        if mode == "batch":
            print(f"Summary: {summary_path}")
        print("=" * 70)

    if failure_count > 0:
        print(f"\n⚠ {failure_count} checkpoints failed. Check log for details.")
        sys.exit(1)
    else:
        if not args.quiet:
            print("\n✓ All checkpoints processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
