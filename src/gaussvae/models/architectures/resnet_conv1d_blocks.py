"""
ResNet-style Conv1D encoder and decoder builders for Gaussian Splatting VAE.

Uses residual connections within convolutional blocks to enable deeper networks
and better gradient flow. Designed for compression tasks where encoder and decoder
are completely separate (no U-Net style skip connections between them).
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def residual_block_1d(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 5,
    dropout_rate: float = 0.1,
    name: str = 'residual_block'
) -> tf.Tensor:
    """
    Residual block for 1D convolutions.
    
    Architecture:
        input → Conv1D → BN → ELU → Dropout → Conv1D → BN → Add(input) → ELU
    
    If input and output channels differ, uses 1x1 projection for residual connection:
        residual = Conv1D(filters=filters, kernel_size=1)(input)
    
    Args:
        x: Input tensor (batch, timesteps, channels)
        filters: Number of output filters
        kernel_size: Convolution kernel size (default: 5)
        dropout_rate: Dropout rate (default: 0.1)
        name: Block name prefix
    
    Returns:
        Output tensor with residual connection applied
    
    Example:
        >>> x = layers.Input(shape=(256, 64))
        >>> x = residual_block_1d(x, filters=128, name='res1')
        >>> # Output shape: (256, 128)
    """
    input_filters = x.shape[-1]
    residual = x
    
    # First conv block
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        name=f'{name}_conv1'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('elu', name=f'{name}_elu1')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_dropout1')(x)
    
    # Second conv block (no activation yet - applied after residual add)
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        name=f'{name}_conv2'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # Project residual if channel dimensions don't match
    if input_filters != filters:
        residual = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same',
            name=f'{name}_residual_projection'
        )(residual)
    
    # Add residual connection
    x = layers.Add(name=f'{name}_add')([x, residual])
    x = layers.Activation('elu', name=f'{name}_elu2')(x)
    
    return x


def build_resnet_conv1d_encoder(
    input_shape: Tuple[int, int] = (512, 8),
    latent_dim: int = 512,
    filters: List[int] = [64, 128, 256, 512, 512],
    kernel_sizes: List[int] = [5, 5, 5, 5, 5],
    pool_sizes: List[Optional[int]] = [2, 2, 2, 2, None],
    dropout_rates: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
    dense_units: int = 1024,
    dense_dropout: float = 0.0,
    name: str = 'encoder'
) -> keras.Model:
    """
    Build ResNet-style Conv1D encoder with residual connections.
    
    Architecture (default params):
    - ResBlock(64, 5) + MaxPool(2) → (256, 64)
    - ResBlock(128, 5) + MaxPool(2) → (128, 128)
    - ResBlock(256, 5) + MaxPool(2) → (64, 256)
    - ResBlock(512, 5) + MaxPool(2) → (32, 512)
    - ResBlock(512, 5) + GlobalAvgPool → (512,)
    - Dense(1024) + BN + ELU → (1024,)
    - Split to z_mean, z_log_var (each latent_dim)
    
    Each ResBlock contains 2 Conv1D layers with residual connection,
    enabling deeper networks without vanishing gradients.
    
    Note: Dropout is disabled by default (all 0.0) for initial training runs.
    Can be enabled later by adjusting dropout_rates and dense_dropout params.
    
    Args:
        input_shape: Input shape (num_gaussians, features_per_gaussian)
        latent_dim: Latent space dimensionality
        filters: List of filter counts per residual block
        kernel_sizes: List of kernel sizes per residual block
        pool_sizes: List of pooling sizes (None = GlobalAvgPool for last)
        dropout_rates: List of dropout rates per block
        dense_units: Units in dense bridge layer before latent
        dense_dropout: Dropout rate for dense bridge layer
        name: Model name
    
    Returns:
        Keras Model with outputs (z_mean, z_log_var)
    
    Example:
        >>> encoder = build_resnet_conv1d_encoder(latent_dim=512)
        >>> z_mean, z_log_var = encoder(input_gaussians)
    """
    if len(filters) != len(kernel_sizes) or len(filters) != len(pool_sizes) or len(filters) != len(dropout_rates):
        raise ValueError("filters, kernel_sizes, pool_sizes, and dropout_rates must have same length")
    
    encoder_input = layers.Input(shape=input_shape, name=f'{name}_input')
    x = encoder_input
    
    # Residual Conv1D blocks with pooling
    for i, (filt, kern, pool, drop) in enumerate(zip(filters, kernel_sizes, pool_sizes, dropout_rates), 1):
        x = residual_block_1d(
            x,
            filters=filt,
            kernel_size=kern,
            dropout_rate=drop,
            name=f'{name}_resblock{i}'
        )
        
        # Pooling: MaxPool or GlobalAvgPool for last block
        if pool is not None:
            x = layers.MaxPooling1D(pool_size=pool, name=f'{name}_pool{i}')(x)
        else:
            x = layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)
    
    # Dense bridge to latent space
    x = layers.Dense(dense_units, activation='elu', name=f'{name}_dense')(x)
    x = layers.BatchNormalization(name=f'{name}_bn_dense')(x)
    x = layers.Dropout(dense_dropout, name=f'{name}_dropout_dense')(x)
    
    # Latent space (NO dropout on these layers)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    encoder = keras.Model(encoder_input, [z_mean, z_log_var], name=name)
    return encoder


def build_resnet_conv1d_decoder(
    latent_dim: int = 512,
    output_shape: Tuple[int, int] = (512, 8),
    filters: List[int] = [512, 512, 256, 128, 64, 32, 16],
    kernel_sizes: List[int] = [5, 5, 5, 5, 5, 5, 5],
    upsample_sizes: List[Optional[int]] = [2, 2, 2, 2, 2, 2, None],
    dropout_rates: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dense_units: List[int] = [1024, 4096],
    reshape_target: Tuple[int, int] = (8, 512),
    name: str = 'decoder'
) -> keras.Model:
    """
    Build ResNet-style Conv1D decoder with residual connections.
    
    Architecture (default params):
    - Dense(1024) + BN + ELU → (1024,)
    - Dense(4096) + BN + ELU → (4096,)
    - Reshape → (8, 512)
    - ResBlock(512, 5) + UpSample(2) → (16, 512)
    - ResBlock(512, 5) + UpSample(2) → (32, 512)
    - ResBlock(256, 5) + UpSample(2) → (64, 256)
    - ResBlock(128, 5) + UpSample(2) → (128, 128)
    - ResBlock(64, 5) + UpSample(2) → (256, 64)
    - ResBlock(32, 5) + UpSample(2) → (512, 32)
    - ResBlock(16, 5) → (512, 16)
    - Conv1D(8, 1) NO activation → (512, 8)
    
    Each ResBlock contains 2 Conv1D layers with residual connection.
    7 residual blocks provide deep convolutional capacity with minimal
    dense layer parameters (75% reduction from original design).
    
    Note: Dropout is disabled by default (all 0.0) for initial training runs.
    Can be enabled later by adjusting dropout_rates param.
    
    Args:
        latent_dim: Latent space dimensionality
        output_shape: Target output shape (num_gaussians, features_per_gaussian)
        filters: List of filter counts per residual block
        kernel_sizes: List of kernel sizes per residual block
        upsample_sizes: List of upsampling sizes (None = no upsampling)
        dropout_rates: List of dropout rates per block
        dense_units: List of dense layer units before reshape
        reshape_target: Shape to reshape dense output to (timesteps, channels)
        name: Model name
    
    Returns:
        Keras Model reconstructing output_shape Gaussians
    
    Example:
        >>> decoder = build_resnet_conv1d_decoder(latent_dim=512)
        >>> reconstruction = decoder(latent_vector)
    """
    if len(filters) != len(kernel_sizes) or len(filters) != len(upsample_sizes) or len(filters) != len(dropout_rates):
        raise ValueError("filters, kernel_sizes, upsample_sizes, and dropout_rates must have same length")
    
    # Validate reshape_target matches dense_units[-1]
    expected_size = reshape_target[0] * reshape_target[1]
    if dense_units[-1] != expected_size:
        raise ValueError(
            f"Last dense_units ({dense_units[-1]}) must equal "
            f"reshape_target product ({reshape_target[0]} * {reshape_target[1]} = {expected_size})"
        )
    
    latent_input = layers.Input(shape=(latent_dim,), name=f'{name}_input')
    x = latent_input
    
    # Progressive expansion from latent via dense layers
    for i, units in enumerate(dense_units, 1):
        x = layers.Dense(units, activation='elu', name=f'{name}_dense{i}')(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i}')(x)
    
    # Reshape to (timesteps, channels) for Conv1D
    x = layers.Reshape(reshape_target, name=f'{name}_reshape')(x)
    
    # Residual Conv1D blocks with optional upsampling
    for i, (filt, kern, upsample, drop) in enumerate(zip(filters, kernel_sizes, upsample_sizes, dropout_rates), 1):
        x = residual_block_1d(
            x,
            filters=filt,
            kernel_size=kern,
            dropout_rate=drop,
            name=f'{name}_resblock{i}'
        )
        
        # Optional upsampling
        if upsample is not None:
            x = layers.UpSampling1D(size=upsample, name=f'{name}_upsample{i}')(x)
    
    # Output layer (NO activation - need unbounded values for Gaussian parameters)
    num_features = output_shape[1]
    decoder_output = layers.Conv1D(
        filters=num_features,
        kernel_size=1,
        padding='same',
        name=f'{name}_output'
    )(x)
    
    decoder = keras.Model(latent_input, decoder_output, name=name)
    return decoder
