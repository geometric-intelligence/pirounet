"""Defines reconstruction loss."""

import torch


def reconstruction_loss(x, x_recon):
    """Compute the reconstruction loss between two sequences.

    This is computed as the mean square errors on the joints'
    positions in 3D.

    Parameters
    ----------
    x :             array
                    Shape = [batch_size, seq_len, input_dim]
                    Input batch of sequences.
    x_recon :       array
                    Shape = [batch_size, seq_len, input_dim]
                    Batch of reconstructed sequences.

    Returns
    ----------
    recon_loss :    array
                    Shape = [batch_size, 1]
                    Reconstruction loss for each sequence
                    in batch.   
    """
    assert x.ndim == x_recon.ndim == 3
    batch_size = x.shape[0]
    recon_loss = (x - x_recon) ** 2

    recon_loss = torch.mean(recon_loss, axis=(1, 2))

    assert recon_loss.shape == (batch_size,)
    return recon_loss