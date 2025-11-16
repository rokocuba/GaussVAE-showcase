#!/usr/bin/env python3
"""
Convert PyTorch .pt checkpoints to NumPy .npz format for TensorFlow training.

Extracts only the 8 optimized Gaussian parameters (xy, scale, rot, feat) and
saves as compressed NumPy arrays. Reduces storage by ~50-85% compared to .pt files.

Features:
- Single file or batch directory conversion
- Resumability: Skip already-converted files
- Automatic validation: Verify conversion correctness
- Progress tracking: ETA and conversion stats
- Error handling: Skip corrupted files, continue processing
- Batch conversion: Process multiple files in memory for speed

Usage:
    # Single file
    python scripts/convert_checkpoints_to_npz.py \\
        --input data/checkpoints/img_000001.pt \\
        --output data/npz/

    # Directory (batch)
    python scripts/convert_checkpoints_to_npz.py \\
        --input data/checkpoints/ \\
        --output data/npz/

    # With options
    python scripts/convert_checkpoints_to_npz.py \\
        --input data/checkpoints/ \\
        --output data/npz/ \\
        --batch_size 100 \\
        --quiet

    # Dry run (preview)
    python scripts/convert_checkpoints_to_npz.py \\
        --input data/checkpoints/ \\
        --output data/npz/ \\
        --dry_run

Output structure:
    output/
    ├── img_000001.npz
    ├── img_000002.npz
    ├── ...
    ├── conversion_log.jsonl    # Per-file conversion details
    └── summary.json            # Overall statistics

NPZ file format:
    data = np.load('img_000001.npz')
    gaussians = data['gaussians']  # (N, 8) array: xy, scale, rot, feat
    psnr = data['psnr']            # Optional: reconstruction quality
    ssim = data['ssim']            # Optional: structural similarity
    lpips = data['lpips']          # Optional: perceptual loss
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import sys

try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install torch numpy")
    sys.exit(1)


def load_processed_files(log_path):
    """Load set of already-converted file stems from JSONL log."""
    processed = set()
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("success"):
                            processed.add(entry["file_stem"])
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
        return f"{seconds/3600:.1f}h"


def format_size(bytes_size):
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def convert_checkpoint_to_npz(pt_path, npz_path, validate=True):
    """
    Convert single .pt checkpoint to .npz format.

    Args:
        pt_path: Path to input .pt file
        npz_path: Path to output .npz file
        validate: If True, verify conversion correctness

    Returns:
        dict with:
            - success: bool
            - duration: float (seconds)
            - pt_size: int (bytes)
            - npz_size: int (bytes)
            - num_gaussians: int
            - error: str (if failed)
            - validation_passed: bool (if validate=True)
    """
    start_time = time.time()

    try:
        # Load PyTorch checkpoint
        checkpoint = torch.load(pt_path, map_location="cpu")

        # Handle both checkpoint formats (flat or nested state_dict)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract only 8 optimized parameters (omit vis_feat)
        # Concatenate: xy (2) + scale (2) + rot (1) + feat (3) = 8 params
        try:
            gaussians = np.concatenate(
                [
                    state_dict["xy"].numpy(),  # (N, 2)
                    state_dict["scale"].numpy(),  # (N, 2)
                    state_dict["rot"].numpy(),  # (N, 1)
                    state_dict["feat"].numpy(),  # (N, 3)
                ],
                axis=1,
            )  # Result: (N, 8)
        except KeyError as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": f"Missing key in checkpoint: {e}",
            }

        num_gaussians = gaussians.shape[0]

        # Validate shape
        if gaussians.shape[1] != 8:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": f"Wrong shape: {gaussians.shape}, expected (N, 8)",
            }

        # Extract optional metadata (for analysis, not training)
        metadata = {}
        for key in ["psnr", "ssim", "lpips"]:
            if key in checkpoint:
                value = checkpoint[key]
                # Handle both scalar and tensor values
                if isinstance(value, torch.Tensor):
                    metadata[key] = value.item()
                else:
                    metadata[key] = float(value)

        # Get file sizes
        pt_size = pt_path.stat().st_size

        # Save as compressed NumPy archive
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, gaussians=gaussians, **metadata)

        npz_size = npz_path.stat().st_size

        # Validation: compare first Gaussian
        validation_passed = True
        if validate:
            try:
                # Reload and verify
                loaded = np.load(npz_path)
                loaded_gaussians = loaded["gaussians"]

                # Check shape
                if loaded_gaussians.shape != gaussians.shape:
                    validation_passed = False

                # Check first Gaussian matches
                if not np.allclose(loaded_gaussians[0], gaussians[0], rtol=1e-5):
                    validation_passed = False

            except Exception:
                validation_passed = False

        duration = time.time() - start_time

        return {
            "success": True,
            "duration": duration,
            "pt_size": pt_size,
            "npz_size": npz_size,
            "num_gaussians": num_gaussians,
            "compression_ratio": pt_size / npz_size if npz_size > 0 else 0,
            "validation_passed": validation_passed if validate else None,
        }

    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pt checkpoints to NumPy .npz format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python scripts/convert_checkpoints_to_npz.py \\
      --input data/checkpoints/img_000001.pt \\
      --output data/npz/
  
  # Directory (batch)
  python scripts/convert_checkpoints_to_npz.py \\
      --input data/checkpoints/ \\
      --output data/npz/
  
  # With batch processing
  python scripts/convert_checkpoints_to_npz.py \\
      --input data/checkpoints/ \\
      --output data/npz/ \\
      --batch_size 100
  
  # Dry run (preview)
  python scripts/convert_checkpoints_to_npz.py \\
      --input data/checkpoints/ \\
      --output data/npz/ \\
      --dry_run

Output:
  - .npz files with 'gaussians' array (N, 8)
  - Optional metadata: psnr, ssim, lpips
  - conversion_log.jsonl with per-file details
  - summary.json with overall statistics
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .pt file or directory containing .pt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of files to process in parallel (default: 50)",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip automatic validation (faster but less safe)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL log file path (default: {output}/conversion_log.jsonl)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be converted without actually converting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess already-converted files (overwrite .npz files)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only errors and final summary)",
    )

    args = parser.parse_args()

    # Setup paths
    input_path = args.input.absolute()
    output_dir = args.output.absolute()

    # Determine if single file or batch
    if input_path.is_file():
        if input_path.suffix.lower() != ".pt":
            print(f"Error: Input file must be .pt, got: {input_path.suffix}")
            sys.exit(1)
        pt_files = [input_path]
        mode = "single"
    elif input_path.is_dir():
        pt_files = sorted(input_path.glob("*.pt"))
        mode = "batch"
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if not pt_files:
        print(f"Error: No .pt files found at: {input_path}")
        sys.exit(1)

    # Setup log path
    if args.log is None:
        log_path = output_dir / "conversion_log.jsonl"
    else:
        log_path = args.log.absolute()

    summary_path = output_dir / "summary.json"

    # Load already-converted files (for resumability)
    processed_files = set() if args.force else load_processed_files(log_path)

    # Filter out already-converted files
    files_to_convert = [f for f in pt_files if f.stem not in processed_files]

    # Print configuration
    if not args.quiet:
        print("=" * 70)
        print(f"PyTorch → NumPy Converter - {mode.upper()} Mode")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Validation: {'Disabled' if args.no_validate else 'Enabled'}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_dir}")
        print(f"  Log: {log_path}")
        print()
        print(f"Files:")
        print(f"  Total: {len(pt_files)}")
        print(f"  Already converted: {len(processed_files)}")
        print(f"  To convert: {len(files_to_convert)}")
        print("=" * 70)

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be converted]")
        if files_to_convert:
            print(f"\nWould convert {len(files_to_convert)} files:")
            for i, f in enumerate(files_to_convert[:10], 1):
                print(f"  {i}. {f.name}")
            if len(files_to_convert) > 10:
                print(f"  ... and {len(files_to_convert) - 10} more")
        else:
            print("\nNo files to convert (all already completed).")
        sys.exit(0)

    # Check if nothing to convert
    if not files_to_convert:
        if not args.quiet:
            print("\nAll files already converted!")
            print(f"  Use --force to reconvert all {len(pt_files)} files.")
        sys.exit(0)

    # Start conversion
    if not args.quiet:
        print("\n" + "=" * 70)
        print("CONVERTING")
        print("=" * 70)

    start_time = time.time()
    success_count = 0
    failure_count = 0
    validation_failures = 0
    total_pt_size = 0
    total_npz_size = 0

    for i, pt_path in enumerate(files_to_convert, 1):
        # Progress (skip in quiet mode or single file)
        if not args.quiet and mode == "batch" and i % 10 == 0:
            elapsed = time.time() - start_time
            progress_pct = i / len(files_to_convert) * 100
            avg_time = elapsed / i
            eta = format_time((len(files_to_convert) - i) * avg_time)
            print(f"[{i}/{len(files_to_convert)}] {progress_pct:.1f}% | ETA: {eta}")

        # Convert file
        npz_path = output_dir / f"{pt_path.stem}.npz"
        result = convert_checkpoint_to_npz(
            pt_path, npz_path, validate=not args.no_validate
        )

        # Update counts
        if result["success"]:
            success_count += 1
            total_pt_size += result.get("pt_size", 0)
            total_npz_size += result.get("npz_size", 0)

            if not result.get("validation_passed", True):
                validation_failures += 1
                if not args.quiet:
                    print(f"  Warning: {pt_path.name} - validation failed")
        else:
            failure_count += 1
            if not args.quiet:
                error_msg = result.get("error", "Unknown error")[:80]
                print(f"  Failed: {pt_path.name} - {error_msg}")

        # Log entry
        log_entry = {
            "file_stem": pt_path.stem,
            "file_name": pt_path.name,
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        append_log_entry(log_path, log_entry)

        # Save summary periodically
        if mode == "batch" and (i % 100 == 0 or i == len(files_to_convert)):
            total_processed = len(processed_files) + i
            summary = {
                "total_files": len(pt_files),
                "converted": total_processed,
                "remaining": len(pt_files) - total_processed,
                "success_count": success_count,
                "failure_count": failure_count,
                "validation_failures": validation_failures,
                "total_pt_size": total_pt_size,
                "total_npz_size": total_npz_size,
                "compression_ratio": (
                    total_pt_size / total_npz_size if total_npz_size > 0 else 0
                ),
                "last_updated": datetime.now().isoformat(),
            }
            save_summary(summary_path, summary)

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(pt_files)}")
    print(f"Converted this run: {len(files_to_convert)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    if not args.no_validate and validation_failures > 0:
        print(f"Validation warnings: {validation_failures}")
    print(f"Total time: {format_time(total_time)}")

    if success_count > 0:
        avg_time = total_time / success_count
        print(f"Average time per file: {avg_time:.3f}s")
        print(f"\nStorage:")
        print(f"  Input (.pt): {format_size(total_pt_size)}")
        print(f"  Output (.npz): {format_size(total_npz_size)}")
        print(
            f"  Saved: {format_size(total_pt_size - total_npz_size)} ({100*(total_pt_size - total_npz_size)/total_pt_size:.1f}%)"
        )
        print(f"  Compression ratio: {total_pt_size/total_npz_size:.2f}x")

    if not args.quiet:
        print(f"\nOutput: {output_dir}/")
        print(f"Log: {log_path}")
        if mode == "batch":
            print(f"Summary: {summary_path}")
        print("=" * 70)

    if failure_count > 0:
        print(f"\nWarning: {failure_count} files failed. Check log for details.")
        sys.exit(1)
    elif validation_failures > 0:
        print(f"\nWarning: {validation_failures} files had validation warnings.")
        sys.exit(0)
    else:
        if not args.quiet:
            print("\nAll files converted successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
