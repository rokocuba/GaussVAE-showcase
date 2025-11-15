"""
Abstract base class for all VAE variants.

Provides common functionality for training, evaluation, and inference.
All VAE models (Conv1D, PointNet, Transformer, etc.) inherit from this.
"""

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras

from .layers import Sampling


class BaseVAE(keras.Model, ABC):
    """
    Abstract base class for Variational Autoencoders.
    
    Provides:
    - Common forward pass logic (encode → sample → decode)
    - Training step with VAE loss (reconstruction + β·KL)
    - Validation step
    - Metrics tracking (total_loss, recon_loss, kl_loss, per-parameter losses)
    - Beta annealing support
    
    Subclasses must implement:
    - build_encoder(): Create encoder architecture
    - build_decoder(): Create decoder architecture
    - compute_reconstruction_loss(): Task-specific reconstruction loss
    
    Example:
        >>> class MyVAE(BaseVAE):
        ...     def build_encoder(self):
        ...         # return encoder model
        ...     def build_decoder(self):
        ...         # return decoder model
        ...     def compute_reconstruction_loss(self, data, reconstruction):
        ...         # return reconstruction loss
        >>> vae = MyVAE(latent_dim=256)
    """
    
    def __init__(self, latent_dim=256, name='vae', **kwargs):
        """
        Initialize base VAE.
        
        Args:
            latent_dim: Dimensionality of latent space
            name: Model name
            **kwargs: Additional arguments passed to keras.Model
        """
        super().__init__(name=name, **kwargs)
        
        self.latent_dim = latent_dim
        
        # Beta for KL divergence weighting (will be updated by callback)
        # Start at 1.0 by default (standard VAE)
        self.beta = tf.Variable(1.0, trainable=False, dtype=tf.float32, name='beta')
        
        # Sampling layer (shared across all VAE variants)
        self.sampling_layer = Sampling()
        
        # Build encoder and decoder (implemented by subclasses)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Metrics trackers
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        
        # Per-parameter loss trackers (for Gaussian Splatting)
        # These can be overridden by subclasses for different data types
        self.loss_xy_tracker = keras.metrics.Mean(name='loss_xy')
        self.loss_scale_tracker = keras.metrics.Mean(name='loss_scale')
        self.loss_rot_tracker = keras.metrics.Mean(name='loss_rot')
        self.loss_feat_tracker = keras.metrics.Mean(name='loss_feat')
    
    @abstractmethod
    def build_encoder(self):
        """
        Build encoder network.
        
        Must return a keras.Model that:
        - Takes input data as input
        - Returns [z_mean, z_log_var] as outputs
        
        Returns:
            encoder: keras.Model with outputs [z_mean, z_log_var]
        """
        pass
    
    @abstractmethod
    def build_decoder(self):
        """
        Build decoder network.
        
        Must return a keras.Model that:
        - Takes latent vector z as input
        - Returns reconstruction as output
        
        Returns:
            decoder: keras.Model
        """
        pass
    
    def compute_reconstruction_loss(self, data, reconstruction):
        """
        Compute reconstruction loss.
        
        Default implementation: MSE for Gaussian Splatting data (512, 8).
        Override this for different data types or loss functions.
        
        Args:
            data: Ground truth data (batch_size, 512, 8)
            reconstruction: Reconstructed data (batch_size, 512, 8)
        
        Returns:
            reconstruction_loss: Scalar tensor
            loss_dict: Dictionary with per-parameter losses (for logging)
        """
        # Per-parameter reconstruction losses (for Gaussian Splatting)
        loss_xy = tf.reduce_mean(tf.square(data[:, :, 0:2] - reconstruction[:, :, 0:2]))
        loss_scale = tf.reduce_mean(tf.square(data[:, :, 2:4] - reconstruction[:, :, 2:4]))
        loss_rot = tf.reduce_mean(tf.square(data[:, :, 4:5] - reconstruction[:, :, 4:5]))
        loss_feat = tf.reduce_mean(tf.square(data[:, :, 5:8] - reconstruction[:, :, 5:8]))
        
        # Total reconstruction loss (equal weights for now)
        reconstruction_loss = loss_xy + loss_scale + loss_rot + loss_feat
        
        # Return both total loss and per-parameter breakdown
        loss_dict = {
            'loss_xy': loss_xy,
            'loss_scale': loss_scale,
            'loss_rot': loss_rot,
            'loss_feat': loss_feat,
        }
        
        return reconstruction_loss, loss_dict
    
    def compute_kl_loss(self, z_mean, z_log_var):
        """
        Compute KL divergence loss.
        
        KL(q(z|x) || p(z)) where p(z) = N(0, I) is the prior.
        
        Args:
            z_mean: Mean of approximate posterior (batch_size, latent_dim)
            z_log_var: Log variance of approximate posterior (batch_size, latent_dim)
        
        Returns:
            kl_loss: Scalar KL divergence
        """
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return kl_loss
    
    def call(self, inputs, return_latent=False, training=None):
        """
        Forward pass through VAE.
        
        Args:
            inputs: Input data
            return_latent: If True, return (reconstruction, z_mean, z_log_var)
                          If False, return only reconstruction
            training: Whether in training mode
        
        Returns:
            If return_latent=False: reconstruction
            If return_latent=True: (reconstruction, z_mean, z_log_var)
        """
        # Encode
        z_mean, z_log_var = self.encoder(inputs, training=training)
        
        # Sample
        z = self.sampling_layer([z_mean, z_log_var])
        
        # Decode
        reconstruction = self.decoder(z, training=training)
        
        if return_latent:
            return reconstruction, z_mean, z_log_var
        return reconstruction
    
    @property
    def metrics(self):
        """Return list of metrics tracked by this model."""
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.loss_xy_tracker,
            self.loss_scale_tracker,
            self.loss_rot_tracker,
            self.loss_feat_tracker,
        ]
    
    def train_step(self, data):
        """
        Custom training step with VAE loss.
        
        Computes: total_loss = reconstruction_loss + beta * kl_loss
        
        Args:
            data: Batch of training data
        
        Returns:
            Dictionary of metric values
        """
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, z_mean, z_log_var = self(data, return_latent=True, training=True)
            
            # Reconstruction loss (per-parameter tracking)
            reconstruction_loss, loss_dict = self.compute_reconstruction_loss(data, reconstruction)
            
            # KL divergence loss
            kl_loss = self.compute_kl_loss(z_mean, z_log_var)
            
            # Total VAE loss with beta weighting
            total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.loss_xy_tracker.update_state(loss_dict['loss_xy'])
        self.loss_scale_tracker.update_state(loss_dict['loss_scale'])
        self.loss_rot_tracker.update_state(loss_dict['loss_rot'])
        self.loss_feat_tracker.update_state(loss_dict['loss_feat'])
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Custom validation step.
        
        Args:
            data: Batch of validation data
        
        Returns:
            Dictionary of metric values
        """
        # Forward pass
        reconstruction, z_mean, z_log_var = self(data, return_latent=True, training=False)
        
        # Reconstruction loss (per-parameter tracking)
        reconstruction_loss, loss_dict = self.compute_reconstruction_loss(data, reconstruction)
        
        # KL divergence loss
        kl_loss = self.compute_kl_loss(z_mean, z_log_var)
        
        # Total VAE loss
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.loss_xy_tracker.update_state(loss_dict['loss_xy'])
        self.loss_scale_tracker.update_state(loss_dict['loss_scale'])
        self.loss_rot_tracker.update_state(loss_dict['loss_rot'])
        self.loss_feat_tracker.update_state(loss_dict['loss_feat'])
        
        return {m.name: m.result() for m in self.metrics}
    
    def encode(self, inputs):
        """
        Encode inputs to latent space.
        
        Args:
            inputs: Input data
        
        Returns:
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution
        """
        return self.encoder(inputs, training=False)
    
    def decode(self, z):
        """
        Decode latent vectors to reconstructions.
        
        Args:
            z: Latent vectors (batch_size, latent_dim)
        
        Returns:
            reconstruction: Decoded data
        """
        return self.decoder(z, training=False)
    
    def sample_latent(self, num_samples=1):
        """
        Sample random latent vectors from prior p(z) = N(0, I).
        
        Useful for generating new samples.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            z: Random latent vectors (num_samples, latent_dim)
        """
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return z
    
    def generate(self, num_samples=1):
        """
        Generate new samples by sampling from prior and decoding.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Generated samples
        """
        z = self.sample_latent(num_samples)
        return self.decode(z)
    
    def reconstruct(self, inputs):
        """
        Reconstruct inputs (encode then decode, using mean of latent).
        
        Uses z_mean instead of sampling for deterministic reconstruction.
        
        Args:
            inputs: Input data
        
        Returns:
            Reconstructed data
        """
        z_mean, _ = self.encode(inputs)
        return self.decode(z_mean)
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
        })
        return config
