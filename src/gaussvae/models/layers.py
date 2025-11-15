"""
Custom layers for VAE models.

Contains reusable layer implementations for all VAE architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """
    Reparameterization trick for VAE latent sampling.
    
    Implements: z = z_mean + exp(0.5 * z_log_var) * epsilon
    where epsilon ~ N(0, I) is random noise for stochasticity.
    
    This allows backpropagation through the sampling operation by
    separating the deterministic part (z_mean, z_log_var) from the
    stochastic part (epsilon).
    
    Example:
        >>> sampling_layer = Sampling()
        >>> z_mean = tf.constant([[0.0, 1.0, -1.0]], dtype=tf.float32)
        >>> z_log_var = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
        >>> z = sampling_layer([z_mean, z_log_var])
        >>> # z will be close to z_mean but with added noise
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        """
        Sample from latent distribution.
        
        Args:
            inputs: List of [z_mean, z_log_var] tensors
                z_mean: (batch_size, latent_dim) - mean of latent distribution
                z_log_var: (batch_size, latent_dim) - log variance of latent distribution
            training: Whether in training mode (unused, kept for API consistency)
        
        Returns:
            z: (batch_size, latent_dim) - sampled latent vector
        """
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        
        # Sample epsilon from standard normal N(0, I)
        epsilon = tf.random.normal(shape=(batch_size, latent_dim), dtype=z_mean.dtype)
        
        # Reparameterization: z = μ + σ * ε  (where σ = exp(0.5 * log_var))
        # Using log_var instead of var for numerical stability
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        """Return layer configuration for serialization."""
        return super().get_config()
    
    def compute_output_shape(self, input_shape):
        """Compute output shape (same as z_mean shape)."""
        z_mean_shape, _ = input_shape
        return z_mean_shape


class KLDivergenceLayer(layers.Layer):
    """
    Optional: Explicit KL divergence computation layer.
    
    Can be added to model as a separate layer that adds KL loss directly.
    Alternative to computing KL in train_step.
    
    Usage:
        >>> kl_layer = KLDivergenceLayer()
        >>> kl_loss = kl_layer([z_mean, z_log_var])
        >>> # kl_loss is a scalar tensor
    
    Note: This is an alternative approach. Current implementation
    computes KL in VAE.train_step() instead.
    """
    
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
    
    def call(self, inputs):
        """
        Compute KL divergence between q(z|x) and p(z).
        
        Args:
            inputs: List of [z_mean, z_log_var]
        
        Returns:
            kl_loss: Scalar KL divergence loss
        """
        z_mean, z_log_var = inputs
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # Formula: -0.5 * sum(1 + log_var - mean^2 - var)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        # Add loss to layer (will be accumulated in model.losses)
        self.add_loss(self.weight * kl_loss)
        
        return kl_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({'weight': self.weight})
        return config


# Future custom layers can be added here
# Examples:
# - AttentionPooling: For PointNet-style architectures
# - SetTransformer: For permutation-invariant processing
# - SpectralNormalization: For stable GAN-like training
# - etc.
