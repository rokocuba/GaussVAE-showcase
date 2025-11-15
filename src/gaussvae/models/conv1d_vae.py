"""
Conv1D VAE implementation for Gaussian Splatting compression.

Uses 1D convolutions on spatially-sorted Gaussians to learn local patterns.
"""

from typing import List, Tuple, Optional
import tensorflow as tf

from .base_vae import BaseVAE
from .architectures.conv1d_blocks import build_conv1d_encoder, build_conv1d_decoder


class Conv1DVAE(BaseVAE):
    """
    Conv1D VAE for spatially-sorted Gaussian Splatting representations.
    
    Uses 1D convolutions to learn local spatial patterns after Morton code sorting.
    This architecture assumes Gaussians have been preprocessed:
    - Sorted by Morton code (spatial proximity)
    - Normalized (per-parameter z-score normalization)
    
    Architecture (default params):
    - Encoder: ~384K params (3 Conv1D blocks + Dense)
    - Decoder: ~1.21M params (2 Dense + 3 Conv1D blocks)
    - Total: ~1.6M params
    
    The encoder progressively downsamples spatial information while increasing
    channel depth (32→64→128), then projects to latent space via dense layers.
    
    The decoder inverts this: dense expansion, reshape to Conv1D-compatible,
    then progressive upsampling with decreasing channels (64→32→16→8).
    
    Args:
        input_shape: Input shape (num_gaussians, features_per_gaussian).
                     Default: (512, 8) for 512 Gaussians with 8 features each
                     (xy, scale, rotation, RGB)
        latent_dim: Latent space dimensionality. Default: 256
        encoder_filters: Conv1D filter sizes for encoder blocks.
                        Default: [32, 64, 128]
        decoder_filters: Conv1D filter sizes for decoder blocks.
                        Default: [64, 32, 16]
        encoder_kernel_sizes: Kernel sizes for encoder Conv1D layers.
                             Default: [5, 5, 5]
        decoder_kernel_sizes: Kernel sizes for decoder Conv1D layers.
                             Default: [5, 5, 5]
        encoder_pool_sizes: Pooling sizes for encoder (None = GlobalAvgPool).
                           Default: [2, 2, None]
        decoder_upsample_sizes: Upsampling sizes for decoder (None = no upsample).
                               Default: [2, 2, None]
        encoder_dropout_rates: Dropout rates per encoder block.
                              Default: [0.1, 0.15, 0.2]
        beta: KL loss weight (can be modified during training via callback).
              Default: 1.0
        **kwargs: Additional arguments passed to BaseVAE
    
    Example:
        >>> vae = Conv1DVAE(latent_dim=256)
        >>> vae.build(input_shape=(None, 512, 8))
        >>> 
        >>> # Encode
        >>> z_mean, z_log_var = vae.encode(batch_data)
        >>> 
        >>> # Reconstruct
        >>> reconstruction = vae.reconstruct(batch_data)
        >>> 
        >>> # Generate new samples
        >>> samples = vae.generate(num_samples=10)
    
    Training:
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, jit_compile=False)
        >>> vae.compile(optimizer=optimizer)
        >>> history = vae.fit(train_dataset, validation_data=dev_dataset, epochs=100)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (512, 8),
        latent_dim: int = 256,
        encoder_filters: List[int] = [32, 64, 128],
        decoder_filters: List[int] = [64, 32, 16],
        encoder_kernel_sizes: List[int] = [5, 5, 5],
        decoder_kernel_sizes: List[int] = [5, 5, 5],
        encoder_pool_sizes: List[Optional[int]] = [2, 2, None],
        decoder_upsample_sizes: List[Optional[int]] = [2, 2, None],
        encoder_dropout_rates: List[float] = [0.1, 0.15, 0.2],
        beta: float = 1.0,
        **kwargs
    ):
        """Initialize Conv1D VAE with architecture parameters."""
        # Store gaussian_shape before calling super().__init__
        # Note: Can't use 'input_shape' - it's a read-only property in keras.Model
        self.gaussian_shape = input_shape
        
        # Store architecture parameters
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.decoder_kernel_sizes = decoder_kernel_sizes
        self.encoder_pool_sizes = encoder_pool_sizes
        self.decoder_upsample_sizes = decoder_upsample_sizes
        self.encoder_dropout_rates = encoder_dropout_rates
        
        # Initialize base VAE (will call build_encoder/build_decoder)
        super().__init__(
            latent_dim=latent_dim,
            **kwargs
        )
        
        # Set beta after initialization (BaseVAE creates it as tf.Variable)
        self.beta.assign(beta)
    
    def build_encoder(self):
        """
        Build Conv1D encoder using conv1d_blocks builder.
        
        Returns:
            Keras Model with outputs (z_mean, z_log_var)
        """
        return build_conv1d_encoder(
            input_shape=self.gaussian_shape,
            latent_dim=self.latent_dim,
            filters=self.encoder_filters,
            kernel_sizes=self.encoder_kernel_sizes,
            pool_sizes=self.encoder_pool_sizes,
            dropout_rates=self.encoder_dropout_rates,
            name='encoder'
        )
    
    def build_decoder(self):
        """
        Build Conv1D decoder using conv1d_blocks builder.
        
        Returns:
            Keras Model reconstructing gaussian_shape Gaussians
        """
        return build_conv1d_decoder(
            latent_dim=self.latent_dim,
            output_shape=self.gaussian_shape,
            filters=self.decoder_filters,
            kernel_sizes=self.decoder_kernel_sizes,
            upsample_sizes=self.decoder_upsample_sizes,
            name='decoder'
        )
    
    def get_config(self):
        """
        Serialize configuration for model saving.
        
        Returns:
            Dictionary with model configuration
        """
        config = super().get_config()
        config.update({
            'encoder_filters': self.encoder_filters,
            'decoder_filters': self.decoder_filters,
            'encoder_kernel_sizes': self.encoder_kernel_sizes,
            'decoder_kernel_sizes': self.decoder_kernel_sizes,
            'encoder_pool_sizes': self.encoder_pool_sizes,
            'decoder_upsample_sizes': self.decoder_upsample_sizes,
            'encoder_dropout_rates': self.encoder_dropout_rates,
        })
        return config
