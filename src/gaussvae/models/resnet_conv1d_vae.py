"""
ResNet-style Conv1D VAE implementation for Gaussian Splatting compression.

Uses residual connections within Conv1D blocks to enable deeper networks
with better gradient flow for improved reconstruction quality.
"""

from typing import List, Tuple, Optional
import tensorflow as tf

from .base_vae import BaseVAE
from .architectures.resnet_conv1d_blocks import build_resnet_conv1d_encoder, build_resnet_conv1d_decoder


class ResNetConv1DVAE(BaseVAE):
    """
    ResNet-style Conv1D VAE for spatially-sorted Gaussian Splatting representations.
    
    Uses residual connections within convolutional blocks to enable deeper networks
    (5-7 layers) without vanishing gradients. Designed for compression tasks where
    the encoder and decoder are completely separate.
    
    This architecture assumes Gaussians have been preprocessed:
    - Sorted by Morton code (spatial proximity)
    - Normalized (per-parameter z-score normalization)
    
    Architecture (default params):
    - Encoder: ~7.0M params (5 ResNet blocks + Dense bridge)
    - Decoder: ~11.5M params (7 ResNet blocks + minimal Dense expansion)
    - Total: ~18.5M params
    
    Key improvements over standard Conv1D VAE:
    - 3x more parameters for higher capacity
    - Residual connections prevent vanishing gradients
    - Deeper convolutional hierarchy learns better features
    - Balanced encoder/decoder (~0.6:1 ratio)
    - Dropout disabled by default for initial training
    
    Args:
        input_shape: Input shape (num_gaussians, features_per_gaussian).
                     Default: (512, 8) for 512 Gaussians with 8 features each
                     (xy, scale, rotation, RGB)
        latent_dim: Latent space dimensionality. Default: 512
        encoder_filters: Conv1D filter sizes for encoder ResNet blocks.
                        Default: [64, 128, 256, 512, 512]
        decoder_filters: Conv1D filter sizes for decoder ResNet blocks.
                        Default: [512, 512, 256, 128, 64, 32, 16]
        encoder_kernel_sizes: Kernel sizes for encoder Conv1D layers.
                             Default: [5, 5, 5, 5, 5]
        decoder_kernel_sizes: Kernel sizes for decoder Conv1D layers.
                             Default: [5, 5, 5, 5, 5, 5, 5]
        encoder_pool_sizes: Pooling sizes for encoder (None = GlobalAvgPool).
                           Default: [2, 2, 2, 2, None]
        decoder_upsample_sizes: Upsampling sizes for decoder (None = no upsample).
                               Default: [2, 2, 2, 2, 2, 2, None]
        encoder_dropout_rates: Dropout rates per encoder block.
                              Default: [0.0, 0.0, 0.0, 0.0, 0.0] (disabled)
        decoder_dropout_rates: Dropout rates per decoder block.
                              Default: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (disabled)
        encoder_dense_units: Units in encoder's dense bridge layer.
                            Default: 1024
        encoder_dense_dropout: Dropout for encoder's dense bridge.
                              Default: 0.0 (disabled)
        decoder_dense_units: List of dense layer units before Conv1D blocks.
                            Default: [1024, 4096]
        decoder_reshape_target: Shape after dense expansion (timesteps, channels).
                               Default: (8, 512)
        beta: KL loss weight (can be modified during training via callback).
              Default: 1.0
        **kwargs: Additional arguments passed to BaseVAE
    
    Example:
        >>> vae = ResNetConv1DVAE(latent_dim=512)
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
        latent_dim: int = 512,
        encoder_filters: List[int] = [64, 128, 256, 512, 512],
        decoder_filters: List[int] = [512, 512, 256, 128, 64, 32, 16],
        encoder_kernel_sizes: List[int] = [5, 5, 5, 5, 5],
        decoder_kernel_sizes: List[int] = [5, 5, 5, 5, 5, 5, 5],
        encoder_pool_sizes: List[Optional[int]] = [2, 2, 2, 2, None],
        decoder_upsample_sizes: List[Optional[int]] = [2, 2, 2, 2, 2, 2, None],
        encoder_dropout_rates: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
        decoder_dropout_rates: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        encoder_dense_units: int = 1024,
        encoder_dense_dropout: float = 0.0,
        decoder_dense_units: List[int] = [1024, 4096],
        decoder_reshape_target: Tuple[int, int] = (8, 512),
        beta: float = 1.0,
        **kwargs
    ):
        """Initialize ResNet Conv1D VAE with architecture parameters."""
        # Store gaussian_shape before calling super().__init__
        # Note: Can't use 'input_shape' - it's a read-only property in keras.Model
        self.gaussian_shape = input_shape
        
        # Store encoder architecture parameters
        self.encoder_filters = encoder_filters
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.encoder_pool_sizes = encoder_pool_sizes
        self.encoder_dropout_rates = encoder_dropout_rates
        self.encoder_dense_units = encoder_dense_units
        self.encoder_dense_dropout = encoder_dense_dropout
        
        # Store decoder architecture parameters
        self.decoder_filters = decoder_filters
        self.decoder_kernel_sizes = decoder_kernel_sizes
        self.decoder_upsample_sizes = decoder_upsample_sizes
        self.decoder_dropout_rates = decoder_dropout_rates
        self.decoder_dense_units = decoder_dense_units
        self.decoder_reshape_target = decoder_reshape_target
        
        # Initialize base VAE (will call build_encoder/build_decoder)
        super().__init__(
            latent_dim=latent_dim,
            **kwargs
        )
        
        # Set beta after initialization (BaseVAE creates it as tf.Variable)
        self.beta.assign(beta)
    
    def build_encoder(self):
        """
        Build ResNet-style Conv1D encoder using resnet_conv1d_blocks builder.
        
        Returns:
            Keras Model with outputs (z_mean, z_log_var)
        """
        return build_resnet_conv1d_encoder(
            input_shape=self.gaussian_shape,
            latent_dim=self.latent_dim,
            filters=self.encoder_filters,
            kernel_sizes=self.encoder_kernel_sizes,
            pool_sizes=self.encoder_pool_sizes,
            dropout_rates=self.encoder_dropout_rates,
            dense_units=self.encoder_dense_units,
            dense_dropout=self.encoder_dense_dropout,
            name='encoder'
        )
    
    def build_decoder(self):
        """
        Build ResNet-style Conv1D decoder using resnet_conv1d_blocks builder.
        
        Returns:
            Keras Model reconstructing gaussian_shape Gaussians
        """
        return build_resnet_conv1d_decoder(
            latent_dim=self.latent_dim,
            output_shape=self.gaussian_shape,
            filters=self.decoder_filters,
            kernel_sizes=self.decoder_kernel_sizes,
            upsample_sizes=self.decoder_upsample_sizes,
            dropout_rates=self.decoder_dropout_rates,
            dense_units=self.decoder_dense_units,
            reshape_target=self.decoder_reshape_target,
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
            'encoder_kernel_sizes': self.encoder_kernel_sizes,
            'encoder_pool_sizes': self.encoder_pool_sizes,
            'encoder_dropout_rates': self.encoder_dropout_rates,
            'encoder_dense_units': self.encoder_dense_units,
            'encoder_dense_dropout': self.encoder_dense_dropout,
            'decoder_filters': self.decoder_filters,
            'decoder_kernel_sizes': self.decoder_kernel_sizes,
            'decoder_upsample_sizes': self.decoder_upsample_sizes,
            'decoder_dropout_rates': self.decoder_dropout_rates,
            'decoder_dense_units': self.decoder_dense_units,
            'decoder_reshape_target': self.decoder_reshape_target,
        })
        return config
