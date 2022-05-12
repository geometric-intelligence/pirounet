"""Architectures of neural networks.

The LstmVAE has been adapted from beyond-imitation, using:
https://towardsdatascience.com/
implementation-differences-in-lstm-layers-tensorflow
-vs-pytorch-77a31d742f74
"""

import logging
import os

import default_config
import models.classifier as classifier
import models.lstm_vae as lstm_vae
import torch
import torch.nn as nn
import utils
from torch.nn import init

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
logging.info(f"Using device {DEVICE}")


class DeepGenerativeModel(lstm_vae.LstmVAE):
    def __init__(
        self,
        n_layers,
        input_features,
        h_features_loop,
        latent_dim,
        output_features,
        seq_len,
        negative_slope,
        label_features,
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
        h_features_loop :
        latent_dim :
        output_features :
        seq_len :
        negative_slope :
        label_features :
        """

        super(DeepGenerativeModel, self).__init__()
        self.label_features = label_features
        self.seq_len = seq_len

        self.encoder = lstm_vae.LstmEncoder(
            n_layers=n_layers,
            input_features=input_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            label_features=label_features,
        )

        self.decoder = lstm_vae.LstmDecoder(
            n_layers=n_layers,
            output_features=output_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            seq_len=seq_len,
            negative_slope=negative_slope,
            label_features=label_features,
        )

        self.classifier = classifier.LinearClassifier(
            input_features, h_features_loop, label_features, seq_len)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        """
            Parameters
            ---
            x : input sequence with shape [batchsize, seq_len, 3*keypoints]
            y : input label with shape [batchsize, 1,label_features]
        """

        y_for_encoder = y.repeat((1, self.seq_len, 1))

        z, z_mu, z_log_var = self.encoder(torch.cat([x, y_for_encoder], dim=2).float())

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        y_for_decoder = y.reshape((y.shape[0], y.shape[-1]))
        x_mu = self.decoder(torch.cat([z, y_for_decoder], dim=1).float())

        return x_mu

    def classify(self, x):
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
        print('Z HAS SHAPE')
        print(z.shape)
        print('z is on device')
        print(z.get_device())
        print('Y HAS SHAPE')
        print(y.shape)
        print(y.get_device())
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


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
        elbo = torch.mean(utils.log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
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
        recon_loss = (x - x_recon) ** 2
        recon_loss = torch.sum(recon_loss, axis=2)
        assert recon_loss.shape == (batch_size, seq_len)

        recon_loss = torch.sum(recon_loss, axis=1)
        assert recon_loss.shape == (batch_size,)
        return recon_loss

    def forward(self, x, y=None, likelihood_func=reconstruction_loss):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = utils.enumerate_discrete(xs, self.model.label_features)
            ys = ys.reshape((ys.shape[0], 1, ys.shape[-1]))
            xs = xs.repeat(self.model.label_features, 1, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -likelihood_func(xs, reconstruction)

        # p(y)
        prior = -torch.squeeze(utils.log_standard_categorical(ys))

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
