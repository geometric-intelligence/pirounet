"""Architectures of LSTM VAE, composed of an encoder and a decoder."

The LstmVAE has been adapted from beyond-imitation, using:
https://towardsdatascience.com/
implementation-differences-in-lstm-layers-tensorflow
-vs-pytorch-77a31d742f74
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import models.losses as losses

class LstmEncoder(nn.Module):
    """Encoder with LSTM layers.

    Parameters
    ----------
    n_layers :  int
                Number of LSTM layers.
    input_dim : int
                Number of features per pose,
                keypoints * 3 dimensions.
    h_dim : int
            Number of nodes in hidden layers.
    latent_dim :    int
                    Dimension of latent space.   
    label_dim : int
                Amount of categorical labels.
    """

    def __init__(
        self,
        n_layers,
        input_dim,
        h_dim,
        latent_dim,
        label_dim,
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


    def reparametrize(self, z_mean, z_logvar):
        """Sample from a multivariate Gaussian.
        The multivariate Gaussian samples in the vector
        space of dimension latent_dim.
        Parameters
        ----------
        z_mean :    array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Mean of the multivariate Gaussian.
        z_logvar :  array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Log of the variance of the multivariate 
                    Gaussian.
        
        Returns
        ----------
        sample :    array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Latent variable sampled from the
                    Gaussian distribution.

        """
        assert z_mean.shape[-1] == self.latent_dim
        assert z_logvar.shape[-1] == self.latent_dim

        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)

        sample = eps.mul(std).add_(z_mean)
        return sample

    def forward(self, inputs):
        """Perform forward pass of the encoder.

        Parameters
        ----------
        inputs :    array
                    Shape = [batch_size, seq_len, input_dim]
                    Input batch of sequence data.
        
        Returns
        ----------
        z_mean :    array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Mean of the latent variable (batch of).
        z_logvar :  array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Log variance of the latent variable (batch of).
        z_sample :  array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Latent variable sampled from the distribution
                    (batch of).
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


class LstmDecoder(nn.Module):
    """Decoder with LSTM layers.
    

    Parameters
    ----------
    n_layers :  int
                Number of LSTM layers.
    output_dim : int
                Number of features per 
                constructed pose,
                keypoints * 3 dimensions.
    h_dim : int
            Number of nodes in hidden layers.
    latent_dim :    int
                    Dimension of latent space. 
    seq_len :   int
                Number of poses in a sequence.  
    neg_slope : int
                Slope for LeakyRelu activation.
    label_dim : int
                Amount of categorical labels.    
    batch_size :    int
                    Amount of examples (sequences)
                    in a  batch.
    """

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
        inputs :    array
                    Shape = [batch_size, latent_dim]
                    Batch of latent variables to be decoded.

        Returns
        ----------
        h : array
            Shape = [batch_size, seq_len, input_dim] (labeled)
            OR
            Shape = [batch_size * label_dim, seq_len, input_dim]
            (unlabeled)
            Output batch of sequences.
        """
        if self.label_dim:
            assert inputs.shape[-1] == self.latent_dim + self.label_dim
        else:
            assert inputs.shape[-1] == self.latent_dim

        h = self.linear(inputs)
        h = self.leakyrelu(h)
        h = h.reshape((h.shape[0], 1, h.shape[-1]))

        h = h.repeat(1, self.seq_len, 1)

        h, (_, _) = self.lstm(h)
        h, (_, _) = self.lstm_out(h)
        return h


class RotationLayer(nn.Module):
    """ Rotate a sequence of poses around axis z.

        Parameters
        ----------
        theta : float
                Angle by which to rotate sequence.    
    """

    def __init__(self, theta):
        super(RotationLayer, self).__init__()
        theta = torch.tensor(theta)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        self.rotation_mat = torch.tensor(
            [[c_theta, -s_theta, 0], [s_theta, c_theta, 0], [0, 0, 1]]
        )

    def forward(self, x):
        """Rotate a minibatch of sequences of poses.

        Parameters
        ----------
        x : array
            Shape = [batch_size, seq_len, input_dim]
            Sequence of poses.

        Returns
        ----------
        x_rot : array
                Shape = [batch_size, seq_len, input_dim]
                Rotated sequence of poses.
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape((batch_size, seq_len, -1, 3))
        x = torch.einsum("...j, ...ij->...i", x, self.rotation_mat)
        x_rot = x.reshape((batch_size, seq_len, -1))
        return x_rot

class LstmVAE(nn.Module):
    """Variational Autoencoder model with LSTM.

    The architecture consists of an (LSTM+encoder)
    and (decoder+LSTM) pair.

    Parameters
    ----------
    n_layers :  int
                Number of LSTM layers.
    input_dim : int
                Number of features per 
                input pose, keypoints * 
                3 dimensions.
    h_dim : int
            Number of nodes in hidden layers.
    latent_dim :    int
                    Dimension of latent space. 
    kl_weight : int
                Weight given to KL divergence
                in regularized loss.
    output_dim :    int
                    Number of features per 
                    output pose, keypoints * 
                    3 dimensions.
    seq_len :   int
                Number of poses in a sequence.  
    neg_slope : int
                Slope for LeakyRelu activation. 
    batch_size :    int
                    Amount of examples (sequences)
                    in a  batch.
    label_dim : int
                Amount of categorical labels.
    with_rotation_layer :   bool
                            Determines use of 
                            rotation layer.

    """

    def __init__(
        self,
        n_layers = 5,
        input_dim = 159,
        h_dim = 100,
        latent_dim = 256,
        kl_weight = 1,
        output_dim = 159,
        seq_len = 40,
        neg_slope = 0,
        batch_size = 80,
        label_dim = 3,
        with_rotation_layer=True
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
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """Perform forward pass of the VAE.

        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        Note : this forward pass is non-conditional, i.e.
        it does not take any labels into account. It is
        not used as part of the Deep Generative model.

        Parameters
        ----------
        x : array
            Shape = [batch_size, seq_len, input_dim]
            Input batch of sequence data.

        Returns
        -------
        x_recon :    array
                    Shape = [batch_size, seq_len, 
                    input_dim]
                    Reconstructed input.
        """
        if self.with_rotation_layer:
            theta = np.random.uniform(0, 2 * np.pi)
            x = RotationLayer(theta)(x)
        z, z_mean, z_log_var = self.encoder(x)

        q_param = (z_mean, z_log_var)

        #self.kl_divergence = losses.kld(q_param)

        x_recon = self.decoder(z)
        if self.with_rotation_layer:
            x_recon = RotationLayer(-theta)(x_recon)
        return x_recon
