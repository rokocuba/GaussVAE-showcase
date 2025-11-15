"""
Configuration management for VAE training.

Supports YAML config files with validation and default values.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Supports both baseline Conv1D and ResNet Conv1D architectures.
    Model selection is determined by which training script you run:
    - train_conv1d_vae.py → Conv1DVAE (baseline)
    - train_resnet_conv1d_vae.py → ResNetConv1DVAE (residual connections)
    
    Baseline Conv1D uses: encoder_filters, decoder_filters, dropout_rates
    ResNet Conv1D uses: All fields below (with sensible defaults in trainer)
    """
    name: str = "conv1d"
    latent_dim: int = 256
    
    # Baseline Conv1D architecture
    encoder_filters: list = field(default_factory=lambda: [32, 64, 128])
    decoder_filters: list = field(default_factory=lambda: [64, 32, 16])
    dropout_rates: list = field(default_factory=lambda: [0.1, 0.15, 0.2])
    
    # ResNet-specific architecture (optional, used by ResNetConv1DVAE)
    encoder_kernel_sizes: Optional[list] = None
    decoder_kernel_sizes: Optional[list] = None
    encoder_pool_sizes: Optional[list] = None
    decoder_upsample_sizes: Optional[list] = None
    encoder_dropout_rates: Optional[list] = None
    decoder_dropout_rates: Optional[list] = None
    encoder_dense_units: Optional[int] = None
    encoder_dense_dropout: Optional[float] = None
    decoder_dense_units: Optional[list] = None
    decoder_reshape_target: Optional[list] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    beta_warmup_epochs: int = 50
    max_beta: float = 1.0  # Maximum beta value after warmup (default: 1.0 for standard VAE)
    mixed_precision: bool = True


@dataclass
class DataConfig:
    """Data paths and loading configuration."""
    train_dir: str = "data/delaunay/npz/train"
    dev_dir: str = "data/delaunay/npz/dev"
    test_dir: str = "data/delaunay/npz/test"
    stats_path: str = "data/normalization_stats.npz"
    shuffle_buffer: int = 1000


@dataclass
class CallbacksConfig:
    """Training callbacks configuration."""
    checkpoint_every: int = 10
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class VAEConfig:
    """Complete VAE training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    output_dir: str = "outputs/vae_training"
    experiment_name: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VAEConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            VAEConfig instance with loaded values

        Example:
            >>> config = VAEConfig.from_yaml('configs/conv1d_base.yaml')
            >>> config.model.latent_dim  # 256
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)

        # Parse nested configs
        model_config = ModelConfig(**yaml_dict.get('model', {}))
        training_config = TrainingConfig(**yaml_dict.get('training', {}))
        data_config = DataConfig(**yaml_dict.get('data', {}))
        callbacks_config = CallbacksConfig(**yaml_dict.get('callbacks', {}))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            callbacks=callbacks_config,
            output_dir=yaml_dict.get('output_dir', 'outputs/vae_training'),
            experiment_name=yaml_dict.get('experiment_name'),
            seed=yaml_dict.get('seed'),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VAEConfig":
        """
        Create configuration from dictionary (for programmatic use).

        Args:
            config_dict: Dictionary with config values

        Returns:
            VAEConfig instance
        """
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        callbacks_config = CallbacksConfig(**config_dict.get('callbacks', {}))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            callbacks=callbacks_config,
            output_dir=config_dict.get('output_dir', 'outputs/vae_training'),
            experiment_name=config_dict.get('experiment_name'),
            seed=config_dict.get('seed'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'callbacks': asdict(self.callbacks),
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
        }

    def save_yaml(self, output_path: str):
        """
        Save configuration to YAML file.

        Useful for saving final config used in training run.

        Args:
            output_path: Where to save YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self):
        """
        Validate configuration values.

        Raises:
            ValueError: If any config values are invalid
        """
        # Validate model
        if self.model.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.model.latent_dim}")
        
        if len(self.model.encoder_filters) == 0:
            raise ValueError("encoder_filters cannot be empty")
        
        if len(self.model.decoder_filters) == 0:
            raise ValueError("decoder_filters cannot be empty")

        # Validate training
        if self.training.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.training.batch_size}")
        
        if self.training.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.training.epochs}")
        
        if self.training.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.training.learning_rate}")
        
        if self.training.beta_warmup_epochs < 0:
            raise ValueError(f"beta_warmup_epochs must be non-negative, got {self.training.beta_warmup_epochs}")

        # Validate data paths exist
        for path_name, path_value in [
            ('train_dir', self.data.train_dir),
            ('dev_dir', self.data.dev_dir),
            ('test_dir', self.data.test_dir),
            ('stats_path', self.data.stats_path),
        ]:
            path = Path(path_value)
            if not path.exists():
                raise FileNotFoundError(f"{path_name} does not exist: {path}")

        # Validate callbacks
        if self.callbacks.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be positive, got {self.callbacks.checkpoint_every}")
        
        if self.callbacks.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.callbacks.early_stopping_patience}")

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["VAEConfig("]
        lines.append(f"  experiment_name={self.experiment_name}")
        lines.append(f"  model: {self.model.name}, latent_dim={self.model.latent_dim}")
        lines.append(f"  training: {self.training.epochs} epochs, batch_size={self.training.batch_size}, lr={self.training.learning_rate}")
        lines.append(f"  data: train_dir={self.data.train_dir}")
        lines.append(f"  output_dir={self.output_dir}")
        lines.append(")")
        return "\n".join(lines)


def load_config(config_path: str) -> VAEConfig:
    """
    Load and validate configuration from YAML file.

    Convenience function that loads and validates in one call.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated VAEConfig instance

    Example:
        >>> config = load_config('configs/conv1d_base.yaml')
        >>> # Config is loaded and validated
    """
    config = VAEConfig.from_yaml(config_path)
    config.validate()
    return config
