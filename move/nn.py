"""Architectures of neural networks.

The LstmVAE has been adapted from beyond-imitation, using:
https://towardsdatascience.com/
implementation-differences-in-lstm-layers-tensorflow
-vs-pytorch-77a31d742f74
"""

import logging
import os
import csv

import classifiers
import default_config
import numpy as np
from collections import OrderedDict
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


class LstmEncoder(torch.nn.Module):
    """Encoder with LSTM layers."""

    def __init__(
        self, 
        n_layers, 
        input_features, 
        h_dim, 
        latent_dim, 
        label_features, 
        bias,
        batch_norm,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_features = input_features
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.label_features = label_features

        if label_features:
            total_input_features = input_features + label_features
        else:
            total_input_features = input_features

        self.lstm = torch.nn.LSTM(
            input_size=total_input_features,
            hidden_size=h_dim,
            batch_first=True,
            num_layers= n_layers
        )

        self.mean_block = torch.nn.Linear(h_dim, latent_dim)
        self.logvar_block = torch.nn.Linear(h_dim, latent_dim)

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
        output_features,
        h_dim,
        latent_dim,
        seq_len,
        negative_slope,
        label_features,
        batch_size,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.output_features = output_features
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.label_features = label_features
        self.batch_size = batch_size

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=negative_slope)

        if label_features:
            input_features_decoder = h_dim + label_features
            total_latent_dim = latent_dim + label_features
        else:
            input_features_decoder = h_dim
            total_latent_dim = latent_dim

        self.linear = torch.nn.Linear(
            total_latent_dim, 
            h_dim
        )

        self.lstm = torch.nn.LSTM(
            input_size=h_dim, 
            hidden_size=h_dim, 
            batch_first=True,
            num_layers=n_layers - 1
        )

        self.lstm_out = torch.nn.LSTM(
            input_size=h_dim, 
            hidden_size=output_features, 
            batch_first=True
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
        if self.label_features:
            assert inputs.shape[-1] == self.latent_dim + self.label_features
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
        ).to(DEVICE)

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
        input_features=3 * 53,
        h_dim=32,
        latent_dim=32,
        kl_weight=0,
        output_features=3 * 53,
        seq_len=128,
        negative_slope=0.2,
        batch_size=8,
        with_rotation_layer=True,
        label_features=None,
    ):
        super(LstmVAE, self).__init__()
        self.latent_dim = latent_dim
        self.with_rotation_layer = with_rotation_layer

        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_features=input_features,
            h_dim=h_dim,
            latent_dim=latent_dim,
            label_features=label_features,
            bias=None,
            batch_norm=None
        )
        self.decoder = LstmDecoder(
            n_layers=n_layers,
            output_features=output_features,
            h_dim=h_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            negative_slope=negative_slope,
            label_features=label_features,
            batch_size=batch_size,
        )
        self.kl_divergence = 0
        self.kl_weight = kl_weight

        assert input_features == output_features

        # This initializes the weights and biases
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """Compute KL-divergence.

        The KL is defined as:
        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                  = -E[log p(z) - log q(z)]

        Formula in Keras was:
        # -0.5*K.mean(K.sum(1 + auto_log_var -
        # K.square(auto_mean) - K.exp(auto_log_var), axis=-1))

        Parameters
        ----------
        z : array-like
            Sample from q-distribuion
        q_param : tuple
            (mu, log_var) of the q-distribution
        p_param: tuple
            (mu, log_var) of the p-distribution

        Returns
        -------
        kl : KL(q||p)
        """
        z_mean, z_logvar = q_param
        kl = -(torch.sum(1 + z_logvar - torch.square(z_mean) - z_logvar.exp(), axis=-1))
        return kl

    def elbo(self, x, x_recon, z, q_param, p_param=None):
        """Compute ELBO.

        Formula in Keras was (reconstruction loss):
        # 0.5*K.mean(K.sum(K.square(auto_input - auto_output), axis=-1))

        Parameters
        ----------
        x : array-like
            Input sequence.
            Shape=[batch_size, seq_len, input_features]
        x_recon : array-like
            Output (reconstructed) sequence.
            Shape=[batch_size, seq_len, output_features]
        z : array-like
            Latent variable. Not a sequence.
            Shape=[batch_size, latent_dim]
        """
        assert x.ndim == x_recon.ndim == 3
        assert z.ndim == 2
        assert z.shape[-1] == self.latent_dim
        batch_size, seq_len, _ = x.shape
        recon_loss = (x - x_recon) ** 2
        recon_loss = torch.sum(recon_loss, axis=2)
        assert recon_loss.shape == (batch_size, seq_len)

        recon_loss = torch.sum(recon_loss)
        regul_loss = self._kld(z, q_param, p_param)
        return recon_loss + self.kl_weight * regul_loss

    def get_recon_loss(self, x, y=None):
        """Perform forward pass of the VAE.

        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters. Records the reconstruction loss of 
        this point.

        Parameters
        ----------
        x : array-like
            Input data.
            Shape=[batch_size, seq_len, input_features].

        y : one hot encoder for Laban effort.

        Returns
        -------
        recon_loss : sum over each absolute distance between
                     each given point in x versus x_recon
        """
        if self.with_rotation_layer:
            theta = np.random.uniform(0, 2 * np.pi)
            x = x.to(DEVICE)
            x = RotationLayer(theta)(x)
        z, z_mean, z_log_var = self.encoder(x)

        q_param = (z_mean, z_log_var)

        self.kl_divergence = self._kld(z, q_param)

        x_recon = self.decoder(z)
        if self.with_rotation_layer:
            x_recon = RotationLayer(-theta)(x_recon)

        batch_size, seq_len, _ = x.shape
        recon_loss = (x - x_recon) ** 2
        recon_loss = torch.sum(recon_loss, axis=2)
        assert recon_loss.shape == (batch_size, seq_len)

        recon_loss = torch.sum(recon_loss)

        return recon_loss

    def forward(self, x, y=None):
        """Perform forward pass of the VAE.

        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        Parameters
        ----------
        x : array-like
            Input data.
            Shape=[batch_size, seq_len, input_features].

        Returns
        -------
        x_mean : array-like
            reconstructed input
        """
        if self.with_rotation_layer:
            theta = np.random.uniform(0, 2 * np.pi)
            x = x.to(DEVICE)
            x = RotationLayer(theta)(x)
        z, z_mean, z_log_var = self.encoder(x)

        q_param = (z_mean, z_log_var)

        self.kl_divergence = self._kld(z, q_param)

        x_recon = self.decoder(z)
        if self.with_rotation_layer:
            x_recon = RotationLayer(-theta)(x_recon)
        return x_recon, z, z_mean, z_log_var


class DeepGenerativeModel(LstmVAE):
    def __init__(
        self,
        n_layers,
        input_features,
        h_dim,
        latent_dim,
        output_features,
        seq_len,
        negative_slope,
        label_features,
        batch_size,
        h_dim_classif,
        neg_slope_classif, 
        n_layers_classif,
        bias,
        batch_norm
    ):

        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.
        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.
        Initialise a new generative model

        Parameters
        ----------
        n_layers :
        input_features :
        h_dim :
        latent_dim :
        output_features :
        seq_len :
        negative_slope :
        label_features :
        """

        super(DeepGenerativeModel, self).__init__()
        self.label_features = label_features
        self.seq_len = seq_len

        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_features=input_features,
            h_dim=h_dim,
            latent_dim=latent_dim,
            label_features=label_features,
            bias=bias,
            batch_norm=batch_norm
        )

        self.decoder = LstmDecoder(
            n_layers=n_layers,
            output_features=output_features,
            h_dim=h_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            negative_slope=negative_slope,
            label_features=label_features,
            batch_size=batch_size,
        )

        self.classifier = classifiers.LinearClassifier(input_features, h_dim_classif, label_features, seq_len, neg_slope_classif, n_layers_classif)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        """
            Parameters
            ---
            x : array-like
                input sequence 
                Shape = [batchsize, seq_len, input_features]
            y : array-like
                input label 
                Shape = [batchsize, 1,label_features]
        """

        y_for_encoder = y.repeat((1, self.seq_len, 1))
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y_for_encoder], dim=2).float())

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        y_for_decoder = y.reshape((y.shape[0], y.shape[-1]))
        x_mu = self.decoder(torch.cat([z, y_for_decoder], dim=1).float())

        return x_mu

    def classify(self, x):
        """
        Classifies input x into logits.

        Parameters
        ----------
        x : array-like
            Shape=[batch_size, seq_len, input_features]

        Returns
        -------
        logits : array-like
                 Shape=[batch_size, label_features]
        """
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()

        x = self.decoder(torch.cat([z, y], dim=1))
        return x
    
    def get_recon_loss(self, x, y=None):
        """Perform forward pass of the VAE.

        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters. Records the reconstruction loss of 
        this point.

        Parameters
        ----------
        x : array-like
            Input data.
            Shape=[batch_size, seq_len, input_features].

        y : array-like
            one hot encoder.
            Shape=[batch_size, label_features].

        Returns
        -------
        recon_loss : sum over each absolute distance between
                     each given point in x versus x_recon
        """
        y_for_encoder = y.repeat((1, self.seq_len, 1))

        z, z_mu, z_log_var = self.encoder(torch.cat([x, y_for_encoder], dim=2).float())

        y_for_decoder = y.reshape((y.shape[0], y.shape[-1]))

        x_mu = self.decoder(torch.cat([z, y_for_decoder], dim=1).float())

        recon_loss = (x - x_mu) ** 2
        recon_loss = torch.sum(recon_loss, axis=2)

        recon_loss = torch.mean(recon_loss)

        return recon_loss    


class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """

    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat((self.mc * self.iw, 1, 1))

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """

    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.sampler = sampler

    def reconstruction_loss(x, x_recon):
        assert x.ndim == x_recon.ndim == 3
        batch_size, seq_len, _ = x.shape
        # print("CHECK FOR NAN IN X AND XRECON")
        # if torch.isnan(x).any():
            # print('x has nan')
        # if torch.isnan(x_recon).any():
            # print('xrecon has nan')
        recon_loss = (x - x_recon) ** 2
        # if torch.isnan(recon_loss).any():
            # print('recon loss ARGUMENT has nan')
        # recon_loss = torch.sum(recon_loss, axis=2)
        recon_loss = torch.mean(recon_loss, axis=(1, 2))
        # if torch.isnan(recon_loss).any():
            # print('recon loss summed over axis 2 has nan')
        # assert recon_loss.shape == (batch_size, seq_len)

        # recon_loss = torch.sum(recon_loss, axis=1)
        # if torch.isnan(recon_loss).any():
            # print('recon loss summed over axis 1 has nan')
        assert recon_loss.shape == (batch_size,)
        return recon_loss

    def forward(self, x, y=None, likelihood_func=reconstruction_loss):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.label_features)
            ys = ys.reshape((ys.shape[0], 1, ys.shape[-1]))
            xs = xs.repeat(self.model.label_features, 1, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        # print('CHECKING IF SAMPLED X AND Y HAVE NANS')
        if torch.isnan(xs).any():
            print('sampled xs has nan')
        if torch.isnan(ys).any():
            print('sampled ys has nan')
        reconstruction = self.model(xs, ys)
        # print('CHECKING IF RECONSTRUCTION HAS NAN')
        if torch.isnan(reconstruction).any():
            print('reconstruction has nan')
        
        # p(x|y,z)
        likelihood = -likelihood_func(xs, reconstruction)
        # print('CHECKING IF LIKELIHOOD HAS NAN')
        if torch.isnan(likelihood).any():
            print('likelihood has nan')

        # p(y)
        prior = -torch.squeeze(log_standard_categorical(ys))
        # print('CHECKING IF PRIOR HAS NAN')
        if torch.isnan(prior).any():
            print('prior has nan')

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - self.model.kl_divergence

        L = self.sampler(elbo)

        if is_labelled:
            return torch.mean(L)

        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H

        return torch.mean(U)


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return (
        torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max
    )


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution with shape [batch_size, 1, label_features]
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=2)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=2)

    return cross_entropy


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.
    Example:
    > x = torch.ones((2, 3))
    > y_dim = 3
    res = enumerate_discrete(x, y_dim) has shape (2*3, 3) and is:
    res = [
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
    ]
    because it assigns both all 3 labels (in one-hot encodings) to
    each of the batch_size elements of x

    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.shape[0]
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())
