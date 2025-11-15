"""
Training orchestration for ResNet Conv1D VAE model.

Handles dataset loading, model creation, callback setup, and training loop
specifically for the ResNet Conv1D VAE architecture with residual connections.

For future architectures (PointNet, Transformer, etc.), create separate
trainer modules (e.g., pointnet_trainer.py, transformer_trainer.py).
"""

from pathlib import Path
from typing import Optional, Dict, Any
import tensorflow as tf

from ..models.resnet_conv1d_vae import ResNetConv1DVAE
from ..data.dataset import create_train_dev_test_datasets
from .config import VAEConfig
from .callbacks import BetaAnnealingCallback


def train_resnet_conv1d_vae(
    config: VAEConfig,
    output_dir: str,
    resume_from: Optional[str] = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Train ResNet Conv1D VAE with given configuration.
    
    Sets up:
    - Model (ResNetConv1DVAE with residual connections and optional mixed precision)
    - Datasets (train/dev/test with preprocessing)
    - Optimizer (Adam with jit_compile=False for Docker compatibility)
    - Callbacks (beta annealing, checkpointing, early stopping, ReduceLR, TensorBoard)
    - Training loop (model.fit)
    
    Args:
        config: VAEConfig with all hyperparameters
        output_dir: Directory to save models and logs (overrides config.output_dir)
        resume_from: Path to checkpoint to resume from (optional)
        verbose: Verbosity level for training (0=silent, 1=progress, 2=debug)
    
    Returns:
        Dictionary with:
            - history: Training history (loss curves, metrics)
            - test_metrics: Final test set performance
            - output_dir: Where outputs were saved
    
    Example:
        >>> config = VAEConfig.from_yaml('configs/resnet_conv1d_512d.yaml')
        >>> results = train_resnet_conv1d_vae(config, output_dir='experiments/run_006')
        >>> print(f"Final test loss: {results['test_metrics']['loss']:.4f}")
    
    Notes:
        - Mixed precision policy is set per-model (not globally) via optimizer dtype
        - Beta annealing prevents posterior collapse during early training
        - Checkpoints save best N models based on validation loss
        - TensorBoard logs written to <output_dir>/logs/
        - ResNet architecture uses ~18.5M params (vs ~2M for baseline Conv1D)
    """
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'checkpoints').mkdir(exist_ok=True)
    (output_path / 'logs').mkdir(exist_ok=True)
    
    # Save config for reproducibility
    config.save_yaml(output_path / 'config.yaml')
    
    if verbose >= 1:
        print("="*60)
        print("ResNet Conv1D VAE Training Setup")
        print("="*60)
        print(f"Output directory: {output_path}")
        print(f"Experiment: {config.experiment_name or 'unnamed'}")
        print(f"Model: {config.model.name}")
        print(f"Latent dim: {config.model.latent_dim}")
        print(f"Encoder filters: {config.model.encoder_filters}")
        print(f"Decoder filters: {config.model.decoder_filters}")
        print(f"Batch size: {config.training.batch_size}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Mixed precision: {config.training.mixed_precision}")
        print("="*60 + "\n")
    
    # Set random seed if provided
    if config.seed is not None:
        tf.random.set_seed(config.seed)
        if verbose >= 1:
            print(f"Random seed set to {config.seed}\n")
    
    # Load datasets
    if verbose >= 1:
        print("Loading datasets...")
    
    train_ds, dev_ds, test_ds = create_train_dev_test_datasets(
        train_dir=config.data.train_dir,
        dev_dir=config.data.dev_dir,
        test_dir=config.data.test_dir,
        stats_path=config.data.stats_path,
        batch_size=config.training.batch_size,
        shuffle_buffer=config.data.shuffle_buffer,
    )
    
    if verbose >= 1:
        print("✓ Datasets loaded\n")
    
    # Create model
    if verbose >= 1:
        print("Creating ResNet Conv1D VAE model...")
    
    # Extract ResNet-specific config (with sensible defaults if not provided)
    encoder_kernel_sizes = getattr(config.model, 'encoder_kernel_sizes', [5] * len(config.model.encoder_filters))
    decoder_kernel_sizes = getattr(config.model, 'decoder_kernel_sizes', [5] * len(config.model.decoder_filters))
    encoder_pool_sizes = getattr(config.model, 'encoder_pool_sizes', [2] * (len(config.model.encoder_filters) - 1) + [None])
    decoder_upsample_sizes = getattr(config.model, 'decoder_upsample_sizes', [2] * (len(config.model.decoder_filters) - 1) + [None])
    encoder_dropout_rates = getattr(config.model, 'encoder_dropout_rates', [0.0] * len(config.model.encoder_filters))
    decoder_dropout_rates = getattr(config.model, 'decoder_dropout_rates', [0.0] * len(config.model.decoder_filters))
    encoder_dense_units = getattr(config.model, 'encoder_dense_units', 1024)
    encoder_dense_dropout = getattr(config.model, 'encoder_dense_dropout', 0.0)
    decoder_dense_units = getattr(config.model, 'decoder_dense_units', [1024, 4096])
    decoder_reshape_target = getattr(config.model, 'decoder_reshape_target', (8, 512))
    
    model = ResNetConv1DVAE(
        input_shape=(512, 8),
        latent_dim=config.model.latent_dim,
        encoder_filters=config.model.encoder_filters,
        decoder_filters=config.model.decoder_filters,
        encoder_kernel_sizes=encoder_kernel_sizes,
        decoder_kernel_sizes=decoder_kernel_sizes,
        encoder_pool_sizes=encoder_pool_sizes,
        decoder_upsample_sizes=decoder_upsample_sizes,
        encoder_dropout_rates=encoder_dropout_rates,
        decoder_dropout_rates=decoder_dropout_rates,
        encoder_dense_units=encoder_dense_units,
        encoder_dense_dropout=encoder_dense_dropout,
        decoder_dense_units=decoder_dense_units,
        decoder_reshape_target=decoder_reshape_target,
        beta=0.0,  # Will be set to proper value by BetaAnnealingCallback at epoch start
        name=config.model.name,
    )
    
    # Build model explicitly for clarity
    model.build(input_shape=(None, 512, 8))
    
    if verbose >= 1:
        print(f"✓ Model created: {model.count_params():,} parameters")
        if verbose >= 2:
            model.summary()
        print()
    
    # Create optimizer with optional mixed precision
    if config.training.mixed_precision:
        # Use mixed precision via optimizer dtype (not global policy)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.training.learning_rate,
            jit_compile=False,  # Avoid libdevice issues in Docker
        )
        # Wrap optimizer for mixed precision
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        if verbose >= 1:
            print("✓ Mixed precision enabled (fp16 compute, fp32 variables)")
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.training.learning_rate,
            jit_compile=False,
        )
        
        if verbose >= 1:
            print("✓ Optimizer: Adam (fp32)")
    
    # Compile model
    model.compile(optimizer=optimizer)
    
    if verbose >= 1:
        print("✓ Model compiled\n")
    
    # Set up callbacks
    if verbose >= 1:
        print("Setting up training callbacks...")
    
    callbacks = [
        # Beta annealing (prevents posterior collapse)
        BetaAnnealingCallback(
            max_beta=config.training.max_beta,
            warmup_epochs=config.training.beta_warmup_epochs,
            verbose=(verbose >= 1),
        ),
        
        # Model checkpointing (save best N models)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / 'checkpoints' / 'vae_epoch_{epoch:03d}'),
            monitor='val_total_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=(1 if verbose >= 1 else 0),
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_total_loss',
            patience=config.callbacks.early_stopping_patience,
            restore_best_weights=True,
            verbose=(1 if verbose >= 1 else 0),
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=config.callbacks.reduce_lr_factor,
            patience=config.callbacks.reduce_lr_patience,
            min_lr=config.callbacks.min_lr,
            verbose=(1 if verbose >= 1 else 0),
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_path / 'logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ),
    ]
    
    if verbose >= 1:
        print("✓ Callbacks configured:")
        print(f"  - Beta annealing: 0 → {config.training.max_beta} over {config.training.beta_warmup_epochs} epochs")
        print(f"  - Early stopping: patience={config.callbacks.early_stopping_patience}")
        print(f"  - ReduceLR: factor={config.callbacks.reduce_lr_factor}, patience={config.callbacks.reduce_lr_patience}")
        print(f"  - TensorBoard logs: {output_path / 'logs'}")
        print()
    
    # Load checkpoint if resuming
    if resume_from:
        if verbose >= 1:
            print(f"Resuming from checkpoint: {resume_from}")
        model.load_weights(resume_from)
        if verbose >= 1:
            print("✓ Checkpoint loaded\n")
    
    # Train
    if verbose >= 1:
        print("="*60)
        print("Starting Training")
        print("="*60 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=config.training.epochs,
        callbacks=callbacks,
        verbose=(2 if verbose >= 2 else 1 if verbose >= 1 else 0),
    )
    
    if verbose >= 1:
        print("\n" + "="*60)
        print("Training Complete")
        print("="*60 + "\n")
    
    # Evaluate on test set
    if verbose >= 0:
        print("Evaluating on test set...")
    
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    
    if verbose >= 1:
        print("✓ Test evaluation complete")
        print(f"  - Test total loss: {test_metrics['total_loss']:.4f}")
        print(f"  - Test reconstruction loss: {test_metrics['recon_loss']:.4f}")
        print(f"  - Test KL loss: {test_metrics['kl_loss']:.4f}")
        print()
    
    # Save final model (use SavedModel format for subclassed models)
    final_model_path = output_path / 'final_model'
    model.save(final_model_path, save_format='tf')
    
    if verbose >= 1:
        print(f"✓ Final model saved: {final_model_path}\n")
        print("="*60)
        print("Training Pipeline Complete")
        print("="*60)
    
    return {
        'history': history.history,
        'test_metrics': test_metrics,
        'output_dir': str(output_path),
    }
