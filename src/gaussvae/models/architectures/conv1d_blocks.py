"""
Conv1D encoder and decoder builders for Gaussian Splatting VAE.

Architecture extracted from notebooks/02_architecture_prototyping.ipynb
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_conv1d_encoder(
    input_shape: Tuple[int, int] = (512, 8),
    latent_dim: int = 256,
    filters: List[int] = [32, 64, 128],
    kernel_sizes: List[int] = [5, 5, 5],
    pool_sizes: List[Optional[int]] = [2, 2, None],
    dropout_rates: List[float] = [0.1, 0.15, 0.2],
    dense_units: int = 512,
    dense_dropout: float = 0.2,
    name: str = 'encoder'
) -> keras.Model:
    """
    Build Conv1D encoder for spatially-sorted Gaussians.
    
    Architecture from notebook (default params):
    - Conv1D(32, 5) + BN + ELU + Dropout(0.1) + MaxPool(2) → (256, 32)
    - Conv1D(64, 5) + BN + ELU + Dropout(0.15) + MaxPool(2) → (128, 64)
    - Conv1D(128, 5) + BN + ELU + Dropout(0.2) + GlobalAvgPool → (128,)
    - Dense(512) + BN + ELU + Dropout(0.2) → (512,)
    - Split to z_mean, z_log_var (each latent_dim)
    
    Args:
        input_shape: Input shape (num_gaussians, features_per_gaussian)
        latent_dim: Latent space dimensionality
        filters: List of Conv1D filter counts per block
        kernel_sizes: List of Conv1D kernel sizes per block
        pool_sizes: List of pooling sizes (None = GlobalAvgPool for last)
        dropout_rates: List of dropout rates per block
        dense_units: Units in dense bridge layer before latent
        dense_dropout: Dropout rate for dense bridge layer
        name: Model name
    
    Returns:
        Keras Model with outputs (z_mean, z_log_var)
    
    Example:
        >>> encoder = build_conv1d_encoder()
        >>> z_mean, z_log_var = encoder(input_gaussians)
    """
    if len(filters) != len(kernel_sizes) or len(filters) != len(pool_sizes) or len(filters) != len(dropout_rates):
        raise ValueError("filters, kernel_sizes, pool_sizes, and dropout_rates must have same length")
    
    encoder_input = layers.Input(shape=input_shape, name=f'{name}_input')
    x = encoder_input
    
    # Conv1D blocks
    for i, (filt, kern, pool, drop) in enumerate(zip(filters, kernel_sizes, pool_sizes, dropout_rates), 1):
        x = layers.Conv1D(
            filters=filt,
            kernel_size=kern,
            padding='same',
            name=f'{name}_conv{i}'
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i}')(x)
        x = layers.Activation('elu', name=f'{name}_elu{i}')(x)
        x = layers.Dropout(drop, name=f'{name}_dropout{i}')(x)
        
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


def build_conv1d_decoder(
    latent_dim: int = 256,
    output_shape: Tuple[int, int] = (512, 8),
    filters: List[int] = [64, 32, 16],
    kernel_sizes: List[int] = [5, 5, 5],
    upsample_sizes: List[Optional[int]] = [2, 2, None],
    dense_units: List[int] = [512, 2048],
    reshape_target: Tuple[int, int] = (128, 16),
    name: str = 'decoder'
) -> keras.Model:
    """
    Build Conv1D decoder to reconstruct Gaussians.
    
    Architecture from notebook (default params):
    - Dense(512) + BN + ELU → (512,)
    - Dense(2048) + BN + ELU → (2048,)
    - Reshape → (128, 16)
    - Conv1D(64, 5) + BN + ELU + UpSample(2) → (256, 64)
    - Conv1D(32, 5) + BN + ELU + UpSample(2) → (512, 32)
    - Conv1D(16, 5) + BN + ELU → (512, 16)
    - Conv1D(8, 1) NO activation → (512, 8)
    
    Args:
        latent_dim: Latent space dimensionality
        output_shape: Target output shape (num_gaussians, features_per_gaussian)
        filters: List of Conv1D filter counts per block
        kernel_sizes: List of Conv1D kernel sizes per block
        upsample_sizes: List of upsampling sizes (None = no upsampling)
        dense_units: List of dense layer units before reshape
        reshape_target: Shape to reshape dense output to (timesteps, channels)
        name: Model name
    
    Returns:
        Keras Model reconstructing output_shape Gaussians
    
    Example:
        >>> decoder = build_conv1d_decoder()
        >>> reconstruction = decoder(latent_vector)
    """
    if len(filters) != len(kernel_sizes) or len(filters) != len(upsample_sizes):
        raise ValueError("filters, kernel_sizes, and upsample_sizes must have same length")
    
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
    
    # Conv1D blocks with optional upsampling
    for i, (filt, kern, upsample) in enumerate(zip(filters, kernel_sizes, upsample_sizes), 1):
        x = layers.Conv1D(
            filters=filt,
            kernel_size=kern,
            padding='same',
            name=f'{name}_conv{i}'
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i + len(dense_units)}')(x)
        x = layers.Activation('elu', name=f'{name}_elu{i}')(x)
        
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
