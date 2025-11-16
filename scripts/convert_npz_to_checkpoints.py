#!/usr/bin/env python3
"""
Convert NumPy .npz files back to PyTorch .pt checkpoint format for Image-GS.

This is the inverse of convert_checkpoints_to_npz.py. It takes the VAE-generated
.npz files and converts them back to the .pt checkpoint format that Image-GS expects.

Features:
- Single file or batch directory conversion
- Resumability: Skip already-converted files
- Automatic validation: Verify conversion correctness
- Progress tracking: ETA and conversion stats
- Error handling: Skip corrupted files, continue processing
- Batch conversion: Process multiple files efficiently

Usage:
    # Single file
    python scripts/convert_npz_to_checkpoints.py \\
        --input reconstructions/img_000001.npz \\
        --output checkpoints/

    # Directory (batch)
    python scripts/convert_npz_to_checkpoints.py \\
        --input reconstructions/ \\
        --output checkpoints/

    # With options
    python scripts/convert_npz_to_checkpoints.py \\
        --input reconstructions/ \\
        --output checkpoints/ \\
        --batch_size 100 \\
        --quiet

    # Dry run (preview)
    python scripts/convert_npz_to_checkpoints.py \\
        --input reconstructions/ \\
        --output checkpoints/ \\
        --dry_run

Output structure:
    output/
    ├── img_000001.pt
    ├── img_000002.pt
    ├── ...
    ├── conversion_log.jsonl    # Per-file conversion details
    └── summary.json            # Overall statistics

Checkpoint file format:
    checkpoint = torch.load('img_000001.pt')
    # Nested state_dict format (Image-GS compatible)
    state_dict = checkpoint['state_dict']
    xy = state_dict['xy']        # (N, 2) - positions
    scale = state_dict['scale']  # (N, 2) - ellipse scales
    rot = state_dict['rot']      # (N, 1) - rotations
    feat = state_dict['feat']    # (N, 3) - RGB colors
    vis_feat = state_dict['vis_feat']  # (N, 3) - visual features (optional)
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


def convert_npz_to_checkpoint(npz_path, pt_path, include_vis_feat=True, validate=True):
    """
    Convert single .npz file to .pt checkpoint format.

    Args:
        npz_path: Path to input .npz file
        pt_path: Path to output .pt file
        include_vis_feat: If True, add zero-filled vis_feat for Image-GS compatibility
        validate: If True, verify conversion correctness

    Returns:
        dict with:
            - success: bool
            - duration: float (seconds)
            - npz_size: int (bytes)
            - pt_size: int (bytes)
            - num_gaussians: int
            - error: str (if failed)
            - validation_passed: bool (if validate=True)
    """
    start_time = time.time()

    try:
        # Load NPZ file
        data = np.load(npz_path)

        if "gaussians" not in data:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": "Missing 'gaussians' key in NPZ file",
            }

        gaussians = data["gaussians"]

        # Validate shape
        if len(gaussians.shape) != 2 or gaussians.shape[1] != 8:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": f"Wrong shape: {gaussians.shape}, expected (N, 8)",
            }

        num_gaussians = gaussians.shape[0]

        # Split gaussians back into components
        # Format: xy (2) + scale (2) + rot (1) + feat (3) = 8 params
        xy = torch.from_numpy(gaussians[:, 0:2].astype(np.float32))
        scale = torch.from_numpy(gaussians[:, 2:4].astype(np.float32))
        rot = torch.from_numpy(gaussians[:, 4:5].astype(np.float32))
        feat = torch.from_numpy(gaussians[:, 5:8].astype(np.float32))

        # Create state_dict
        state_dict = {
            "xy": xy,
            "scale": scale,
            "rot": rot,
            "feat": feat,
        }

        # Add vis_feat if requested (Image-GS needs this, initialized to zeros)
        if include_vis_feat:
            vis_feat = torch.zeros(num_gaussians, 3, dtype=torch.float32)
            state_dict["vis_feat"] = vis_feat

        # Create checkpoint in nested format (Image-GS compatible)
        checkpoint = {"state_dict": state_dict}

        # Add optional metadata if present
        for key in ["psnr", "ssim", "lpips"]:
            if key in data:
                checkpoint[key] = float(data[key])

        # Get file sizes
        npz_size = npz_path.stat().st_size

        # Save as PyTorch checkpoint
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, pt_path)

        pt_size = pt_path.stat().st_size

        # Validation: reload and compare first Gaussian
        validation_passed = True
        if validate:
            try:
                # Reload and verify
                loaded = torch.load(pt_path, map_location="cpu")
                loaded_state_dict = loaded.get("state_dict", loaded)

                # Check keys present
                required_keys = ["xy", "scale", "rot", "feat"]
                if not all(k in loaded_state_dict for k in required_keys):
                    validation_passed = False

                # Check first Gaussian matches
                loaded_gaussians = torch.cat(
                    [
                        loaded_state_dict["xy"],
                        loaded_state_dict["scale"],
                        loaded_state_dict["rot"],
                        loaded_state_dict["feat"],
                    ],
                    dim=1,
                )

                if not torch.allclose(
                    loaded_gaussians[0],
                    torch.from_numpy(gaussians[0].astype(np.float32)),
                    rtol=1e-5,
                ):
                    validation_passed = False

            except Exception:
                validation_passed = False

        duration = time.time() - start_time

        return {
            "success": True,
            "duration": duration,
            "npz_size": npz_size,
            "pt_size": pt_size,
            "num_gaussians": num_gaussians,
            "expansion_ratio": pt_size / npz_size if npz_size > 0 else 0,
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
        description="Convert NumPy .npz files back to PyTorch .pt checkpoint format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python scripts/convert_npz_to_checkpoints.py \\
      --input reconstructions/img_000001.npz \\
      --output checkpoints/
  
  # Directory (batch)
  python scripts/convert_npz_to_checkpoints.py \\
      --input reconstructions/ \\
      --output checkpoints/
  
  # With batch processing
  python scripts/convert_npz_to_checkpoints.py \\
      --input reconstructions/ \\
      --output checkpoints/ \\
      --batch_size 100
  
  # Dry run (preview)
  python scripts/convert_npz_to_checkpoints.py \\
      --input reconstructions/ \\
      --output checkpoints/ \\
      --dry_run

Output:
  - .pt files with nested state_dict format (Image-GS compatible)
  - conversion_log.jsonl with per-file details
  - summary.json with overall statistics
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .npz file or directory containing .npz files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of files to process in parallel (default: 50)",
    )
    parser.add_argument(
        "--no_vis_feat",
        action="store_true",
        help="Don't add vis_feat to checkpoint (may break Image-GS)",
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
        help="Reprocess already-converted files (overwrite .pt files)",
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
        if input_path.suffix.lower() != ".npz":
            print(f"Error: Input file must be .npz, got: {input_path.suffix}")
            sys.exit(1)
        npz_files = [input_path]
        mode = "single"
    elif input_path.is_dir():
        npz_files = sorted(input_path.glob("*.npz"))
        mode = "batch"
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if not npz_files:
        print(f"Error: No .npz files found at: {input_path}")
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
    files_to_convert = [f for f in npz_files if f.stem not in processed_files]

    # Print configuration
    if not args.quiet:
        print("=" * 70)
        print(f"NumPy → PyTorch Converter - {mode.upper()} Mode")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Validation: {'Disabled' if args.no_validate else 'Enabled'}")
        print(
            f"  Include vis_feat: {'No (WARNING!)' if args.no_vis_feat else 'Yes'}"
        )
        print(f"  Input: {input_path}")
        print(f"  Output: {output_dir}")
        print(f"  Log: {log_path}")
        print()
        print(f"Files:")
        print(f"  Total: {len(npz_files)}")
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
        print("\nAll files already converted!")
        if not args.quiet:
            print(f"  Use --force to reconvert all {len(npz_files)} files.")
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
    total_npz_size = 0
    total_pt_size = 0

    for i, npz_path in enumerate(files_to_convert, 1):
        # Progress (skip in quiet mode or single file)
        if not args.quiet and mode == "batch" and i % 10 == 0:
            elapsed = time.time() - start_time
            progress_pct = i / len(files_to_convert) * 100
            avg_time = elapsed / i
            eta = format_time((len(files_to_convert) - i) * avg_time)
            print(f"[{i}/{len(files_to_convert)}] {progress_pct:.1f}% | ETA: {eta}")

        # Convert file
        pt_path = output_dir / f"{npz_path.stem}.pt"
        result = convert_npz_to_checkpoint(
            npz_path,
            pt_path,
            include_vis_feat=not args.no_vis_feat,
            validate=not args.no_validate,
        )

        # Update counts
        if result["success"]:
            success_count += 1
            total_npz_size += result.get("npz_size", 0)
            total_pt_size += result.get("pt_size", 0)

            if not result.get("validation_passed", True):
                validation_failures += 1
                if not args.quiet:
                    print(f"  Warning: {npz_path.name} - validation failed")
        else:
            failure_count += 1
            if not args.quiet:
                error_msg = result.get("error", "Unknown error")[:80]
                print(f"  Failed: {npz_path.name} - {error_msg}")

        # Log entry
        log_entry = {
            "file_stem": npz_path.stem,
            "file_name": npz_path.name,
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        append_log_entry(log_path, log_entry)

        # Save summary periodically
        if mode == "batch" and (i % 100 == 0 or i == len(files_to_convert)):
            total_processed = len(processed_files) + i
            summary = {
                "total_files": len(npz_files),
                "converted": total_processed,
                "remaining": len(npz_files) - total_processed,
                "success_count": success_count,
                "failure_count": failure_count,
                "validation_failures": validation_failures,
                "total_npz_size": total_npz_size,
                "total_pt_size": total_pt_size,
                "expansion_ratio": (
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
    print(f"Total files: {len(npz_files)}")
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
        print(f"  Input (.npz): {format_size(total_npz_size)}")
        print(f"  Output (.pt): {format_size(total_pt_size)}")
        print(
            f"  Increase: {format_size(total_pt_size - total_npz_size)} ({100*(total_pt_size - total_npz_size)/total_npz_size:.1f}%)"
        )
        print(f"  Expansion ratio: {total_pt_size/total_npz_size:.2f}x")

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
        print("\nAll files converted successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
