"""
Custom Keras callbacks for VAE training.
"""

import tensorflow as tf
from tensorflow import keras


class BetaAnnealingCallback(keras.callbacks.Callback):
    """
    Linearly increase beta from 0 to max_beta over warmup_epochs.
    
    Beta controls the KL divergence loss weight in the VAE objective:
        total_loss = reconstruction_loss + beta * kl_loss
    
    - beta = 0: Pure autoencoder (no regularization)
    - beta = 1: Standard VAE (full KL regularization)
    
    Gradual beta warmup prevents "posterior collapse" where the model
    learns to ignore the latent space entirely.
    
    Args:
        max_beta: Final beta value after warmup. Default: 1.0
        warmup_epochs: Number of epochs to reach max_beta. Default: 50
        verbose: Whether to print beta updates each epoch. Default: True
    
    Example:
        >>> callback = BetaAnnealingCallback(max_beta=1.0, warmup_epochs=50)
        >>> model.fit(train_ds, epochs=100, callbacks=[callback])
        
        Epoch 1: beta = 0.02
        Epoch 25: beta = 0.50
        Epoch 50: beta = 1.00
        Epoch 75: beta = 1.00 (stays at max)
    
    Note:
        The model must have a `beta` attribute (tf.Variable) for this
        callback to work. BaseVAE and its subclasses have this by default.
    """
    
    def __init__(
        self,
        max_beta: float = 1.0,
        warmup_epochs: int = 50,
        verbose: bool = True
    ):
        """Initialize beta annealing callback."""
        super().__init__()
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Update beta at the start of each epoch.
        
        Beta increases linearly: beta = max_beta * (epoch + 1) / warmup_epochs
        After warmup_epochs, beta stays at max_beta.
        
        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dictionary of metrics (unused)
        """
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → max_beta
            beta = self.max_beta * (epoch + 1) / self.warmup_epochs
        else:
            # After warmup: stay at max_beta
            beta = self.max_beta
        
        # Update model's beta variable
        if hasattr(self.model, 'beta'):
            self.model.beta.assign(beta)
        else:
            raise AttributeError(
                f"Model {self.model.__class__.__name__} does not have a 'beta' attribute. "
                "Make sure your model inherits from BaseVAE or has a beta tf.Variable."
            )
        
        # Print update if verbose
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: beta = {beta:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Log beta value to TensorBoard at end of epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dictionary of metrics (will add 'beta' key)
        """
        if logs is not None and hasattr(self.model, 'beta'):
            # Add beta to logs for TensorBoard/history tracking
            logs['beta'] = float(self.model.beta.numpy())
