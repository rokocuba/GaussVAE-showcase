"""
Conv1D architecture builders for Gaussian VAE.
"""

from .conv1d_blocks import build_conv1d_encoder, build_conv1d_decoder

__all__ = ['build_conv1d_encoder', 'build_conv1d_decoder']
