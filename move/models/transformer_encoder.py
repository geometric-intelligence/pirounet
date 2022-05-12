 import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
logging.info(f"Using device {DEVICE}")
 
 class Transformer(nn.Module):
    """Encoder with LSTM layers."""

    def __init__(
        self, n_layers, n_t_layers, input_features, h_features_loop, latent_dim, label_features=None
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_t_layers = n_t_layers
        self.input_features = input_features
        self.h_features_loop = h_features_loop
        self.latent_dim = latent_dim
        self.label_features = label_features

        if label_features:
            total_input_features = input_features + label_features
        else:
            total_input_features = input_features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers = n_t_layers
        )

        self.mean_block = torch.nn.Linear(h_features_loop, latent_dim)
        self.logvar_block = torch.nn.Linear(h_features_loop, latent_dim)

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
            shape=[batch_size, seq_len, input_features]
            where input_features = 3 * n_body_joints.
        """

        if self.label_features:
            assert inputs.shape[-1] == self.input_features + self.label_features
        else:
            assert inputs.shape[-1] == self.input_features

        batch_size, seq_len, _ = inputs.shape

        logging.debug(f"- Encoder inputs of shape {inputs.shape}")

        h, (h_last_t, c_last_t) = self.transformer(inputs)
        logging.debug(
            f"Transformer gives h of shape {h.shape} & h_last_t of shape {h_last_t.shape} "
        )

        for i in range(self.n_layers - 1):
            logging.debug(f"Transformer loop iteration {i}/{self.n_layers-1}.")
            h, (h_last_t, c_last_t) = self.transformer(h, (h_last_t, c_last_t))
            assert h.shape == (batch_size, seq_len, self.h_features_loop)
            assert h_last_t.shape == (1, batch_size, self.h_features_loop)

        logging.debug("Computing encoder output.")
        h1_last_t = h_last_t.squeeze(axis=0)
        assert h1_last_t.shape == (batch_size, self.h_features_loop)
        z_mean = self.mean_block(h1_last_t)
        z_logvar = self.logvar_block(h1_last_t)
        z_sample = self.reparametrize(z_mean, z_logvar)

        logging.debug("Encoder done.")
        return z_sample, z_mean, z_logvar

    def forward(self, x_in, x_lengths, apply_softmax=False):

        # Embed
        x_in = self.embeddings(x_in)

        # Feed into RNN
        out, h_n = self.LSTM(x_in) #shape of out: T*N*D

        # Gather the last relevant hidden state
        out = out[-1,:,:] # N*D

        # FC layers
        z = self.dropout(out)
        z = self.fc1(z)
        z = self.dropout(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred