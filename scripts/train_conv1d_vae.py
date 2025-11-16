#!/usr/bin/env python
"""
Train Conv1D VAE from command line.

Example:
    python scripts/train_conv1d_vae.py --config configs/conv1d_base.yaml --output experiments/run_001
    
    # Resume from checkpoint
    python scripts/train_conv1d_vae.py --config configs/conv1d_base.yaml --output experiments/run_001 \
        --resume experiments/run_001/checkpoints/vae_epoch_050_loss_0.1234.keras
    
    # Silent mode
    python scripts/train_conv1d_vae.py --config configs/conv1d_base.yaml --output experiments/run_001 --verbose 0
"""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gaussvae.training.config import VAEConfig
from gaussvae.training.conv1d_trainer import train_conv1d_vae


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Conv1D Gaussian Splatting VAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train_conv1d_vae.py --config configs/conv1d_base.yaml --output experiments/run_001
  
  # Resume from checkpoint
  python scripts/train_conv1d_vae.py --config configs/conv1d_base.yaml --output experiments/run_001 \
      --resume experiments/run_001/checkpoints/vae_epoch_050_loss_0.1234.keras
  
  # Debug mode with detailed logging
  python scripts/train_conv1d_vae.py --config configs/conv1d_base.yaml --output experiments/run_001 --verbose 2
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (required)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for models and logs (overrides config.output_dir if provided)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level: 0=silent, 1=progress (default), 2=debug'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1
    
    # Load config
    if args.verbose >= 1:
        print(f"Loading config from {args.config}")
    
    try:
        config = VAEConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    
    # Override output_dir if provided via CLI
    output_dir = args.output if args.output is not None else config.output_dir
    
    # Validate config
    try:
        config.validate()
    except Exception as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        print("\nPlease check your config file for errors.", file=sys.stderr)
        return 1
    
    if args.verbose >= 1:
        print("✓ Config loaded and validated\n")
    
    # Validate resume checkpoint if provided
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume checkpoint not found: {resume_path}", file=sys.stderr)
            return 1
    
    # Train
    try:
        if args.verbose >= 1:
            print(f"Starting Conv1D VAE training, output to {output_dir}\n")
        
        results = train_conv1d_vae(
            config=config,
            output_dir=output_dir,
            resume_from=args.resume,
            verbose=args.verbose
        )
        
        # Print summary
        if args.verbose >= 1:
            print("\n" + "="*60)
            print("Training Complete!")
            print("="*60)
            print(f"Output directory: {results['output_dir']}")
            print(f"Final test loss: {results['test_metrics']['loss']:.4f}")
            print(f"Final test recon loss: {results['test_metrics']['reconstruction_loss']:.4f}")
            print(f"Final test KL loss: {results['test_metrics']['kl_loss']:.4f}")
            print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}", file=sys.stderr)
        
        if args.verbose >= 2:
            import traceback
            print("\nFull traceback:", file=sys.stderr)
            traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
