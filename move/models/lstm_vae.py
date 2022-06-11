"""Architectures of LSTM VAE.

The LstmVAE has been adapted from beyond-imitation, using:
https://towardsdatascience.com/
implementation-differences-in-lstm-layers-tensorflow
-vs-pytorch-77a31d742f74
"""

import models.classifiers as classifiers
import models.losses as losses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

class LstmEncoder(torch.nn.Module):
    """Encoder with LSTM layers."""

    def __init__(
        self,
        n_layers,
        input_dim,
        h_dim,
        latent_dim,
        label_dim,
        bias,
        batch_norm,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        if label_dim:
            total_input_dim = input_dim + label_dim
        else:
            total_input_dim = input_dim

        self.lstm = torch.nn.LSTM(
            input_size=total_input_dim,
            hidden_size=h_dim,
            batch_first=True,
            num_layers=n_layers,
        )

        self.mean_block = torch.nn.Linear(h_dim, latent_dim)
        self.logvar_block = torch.nn.Linear(h_dim, latent_dim)

        # self.leakyrelu = torch.nn.LeakyReLU(negative_slope=neg_slope)

    def reparametrize(self, z_mean, z_logvar):
        """Sample from a multivariate Gaussian.
        The multivariate Gaussian samples in the vector
        space of dimension latent_dim.
        Parameters
        ----------
        z_mean : array-like, shape=[batch_size, latent_dim]
            Mean of the multivariate Gaussian.
        z_logvar : array-like, shape=[batch_size, latent_dim]
            Log of the variance of the multivariate Gaussian.
        """
        assert z_mean.shape[-1] == self.latent_dim
        assert z_logvar.shape[-1] == self.latent_dim

        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(z_mean)

    def forward(self, inputs):
        """Perform forward pass of the encoder.

        Parameters
        ----------
        inputs : array-like
            shape=[batch_size, seq_len, input_dim]
            where input_dim = 3 * n_body_joints.
        """

        if self.label_dim:
            assert inputs.shape[-1] == self.input_dim + self.label_dim
        else:
            assert inputs.shape[-1] == self.input_dim

        batch_size, seq_len, _ = inputs.shape

        h, (h_last_t_all, c_last_t) = self.lstm(inputs)

        assert h.shape == (batch_size, seq_len, self.h_dim)
        assert h_last_t_all.shape == (self.n_layers, batch_size, self.h_dim)
        h_last_t = h_last_t_all[self.n_layers - 1, :, :]
        assert h_last_t.shape == (batch_size, self.h_dim)

        z_mean = self.mean_block(h_last_t)
        z_logvar = self.logvar_block(h_last_t)
        z_sample = self.reparametrize(z_mean, z_logvar)

        return z_sample, z_mean, z_logvar


class LstmDecoder(torch.nn.Module):
    """Decoder with LSTM layers."""

    def __init__(
        self,
        n_layers,
        output_dim,
        h_dim,
        latent_dim,
        seq_len,
        neg_slope,
        label_dim,
        batch_size,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.label_dim = label_dim
        self.batch_size = batch_size

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=neg_slope)

        if label_dim:
            total_latent_dim = latent_dim + label_dim
        else:
            total_latent_dim = latent_dim

        self.linear = torch.nn.Linear(total_latent_dim, h_dim)

        self.lstm = torch.nn.LSTM(
            input_size=h_dim,
            hidden_size=h_dim,
            batch_first=True,
            num_layers=n_layers - 1,
        )

        self.lstm_out = torch.nn.LSTM(
            input_size=h_dim, hidden_size=output_dim, batch_first=True
        )

    def forward(self, inputs):
        """Perform forward pass of the decoder.

        This one also perform a copy of the initial state
        https://curiousily.com/posts/time-series-anomaly
        -detection-using-lstm-autoencoder-with-pytorch-in-python/

        Parameters
        ----------
        inputs : array-like
            Shape=[batch_size, latent_dim]
        """
        if self.label_dim:
            assert inputs.shape[-1] == self.latent_dim + self.label_dim
        else:
            assert inputs.shape[-1] == self.latent_dim

        h = self.linear(inputs)
        h = self.leakyrelu(h)
        h = h.reshape((h.shape[0], 1, h.shape[-1]))

        h = h.repeat(1, self.seq_len, 1)

        h, (h_last_t, c_last_t) = self.lstm(h)
        h, (h_last_t, c_last_t) = self.lstm_out(h)
        return h


class RotationLayer(torch.nn.Module):
    """Rotate a sequence of skeletons around axis z."""

    def __init__(self, theta):
        super(RotationLayer, self).__init__()
        theta = torch.tensor(theta)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        self.rotation_mat = torch.tensor(
            [[c_theta, -s_theta, 0], [s_theta, c_theta, 0], [0, 0, 1]]
        )

    def forward(self, x):
        """Rotate a minibatch of sequences of skeletons.

        Parameters
        ----------
        x : array-like
            Sequence of skeletons.
            Shape=[batch_size, seq_len, 3*n_body_joints]
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape((batch_size, seq_len, -1, 3))
        x = torch.einsum("...j, ...ij->...i", x, self.rotation_mat)
        return x.reshape((batch_size, seq_len, -1))


class LstmVAE(torch.nn.Module):
    """Variational Autoencoder model with LSTM.

    The architecture consists of an (LSTM+encoder)
    and (decoder+LSTM) pair.
    """

    def __init__(
        self,
        n_layers=2,
        input_dim=3 * 53,
        h_dim=32,
        latent_dim=32,
        kl_weight=0,
        output_dim=3 * 53,
        seq_len=128,
        neg_slope=0.2,
        batch_size=8,
        with_rotation_layer=True,
        label_dim=None,
    ):
        super(LstmVAE, self).__init__()
        self.latent_dim = latent_dim
        self.with_rotation_layer = with_rotation_layer

        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_dim=input_dim,
            h_dim=h_dim,
            latent_dim=latent_dim,
            label_dim=label_dim,
            bias=None,
            batch_norm=None,
        )
        self.decoder = LstmDecoder(
            n_layers=n_layers,
            output_dim=output_dim,
            h_dim=h_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            neg_slope=neg_slope,
            label_dim=label_dim,
            batch_size=batch_size,
        )
        self.kl_divergence = 0
        self.kl_weight = kl_weight

        assert input_dim == output_dim

        # This initializes the weights and biases
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """Perform forward pass of the VAE.

        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        Parameters
        ----------
        x : array-like
            Input data.
            Shape=[batch_size, seq_len, input_dim].

        Returns
        -------
        x_mean : array-like
            reconstructed input
        """
        if self.with_rotation_layer:
            theta = np.random.uniform(0, 2 * np.pi)
            x = RotationLayer(theta)(x)
        z, z_mean, z_log_var = self.encoder(x)

        q_param = (z_mean, z_log_var)

        self.kl_divergence = losses.kld(z, q_param)

        x_recon = self.decoder(z)
        if self.with_rotation_layer:
            x_recon = RotationLayer(-theta)(x_recon)
        return x_recon, z, z_mean, z_log_var
