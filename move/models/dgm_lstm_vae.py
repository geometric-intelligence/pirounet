"""Architectures of DGM LSTM VAE."""
import os
import models.losses as losses
import models.utils as utils
import torch.nn
import torch.nn.functional as F
from models.classifiers import ActorClassifier, LinearClassifier
from models.lstm_vae import LstmDecoder, LstmEncoder, LstmVAE
from torch.autograd import Variable


class DeepGenerativeModel(LstmVAE):
    def __init__(
        self,
        n_layers,
        input_dim,
        h_dim,
        latent_dim,
        output_dim,
        seq_len,
        neg_slope,
        label_dim,
        batch_size,
        h_dim_classif,
        neg_slope_classif,
        n_layers_classif,
        bias,
        batch_norm,
        classifier
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
        input_dim :
        h_dim :
        latent_dim :
        output_dim :
        seq_len :
        neg_slope :
        label_dim :
        """

        super(DeepGenerativeModel, self).__init__()
        self.label_dim = label_dim
        self.seq_len = seq_len

        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_dim=input_dim,
            h_dim=h_dim,
            latent_dim=latent_dim,
            label_dim=label_dim,
            bias=bias,
            batch_norm=batch_norm,
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

        if classifier == 'linear':
            self.classifier = LinearClassifier(
                input_dim=input_dim,
                h_dim=h_dim_classif,
                label_dim=label_dim,
                seq_len=seq_len,
                neg_slope=neg_slope_classif,
                n_layers=n_layers_classif,
            )

        if classifier == 'transformer':
            self.classifier = ActorClassifier(
                seq_len=seq_len,
                label_dim=label_dim,
                input_dim=input_dim,
                embed_dim=16,
                dim_feedforward=2,
            )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        """
        Parameters
        ---
        x : array-like
            input sequence
            Shape = [batchsize, seq_len, input_dim]
        y : array-like
            input label
            Shape = [batchsize, 1,label_dim]
        """

        y_for_encoder = y.repeat((1, self.seq_len, 1))
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y_for_encoder], dim=2).float())

        self.kl_divergence = losses.kld(z, (z_mu, z_log_var))

        y_for_decoder = y.reshape((y.shape[0], y.shape[-1]))
        x_mu = self.decoder(torch.cat([z, y_for_decoder], dim=1).float())

        return x_mu

    def classify(self, x):
        """Classify input x into logits.

        Parameters
        ----------
        x : array-like
            Shape=[batch_size, seq_len, input_dim]

        Returns
        -------
        logits : array-like
                 Shape=[batch_size, label_dim]
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


class SVI(torch.nn.Module):
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

    def forward(self, x, y=None, likelihood_func=losses.reconstruction_loss):
        batch_size = x.shape[0]
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = utils.enumerate_discrete(xs, self.model.label_dim)
            ys = ys.reshape((ys.shape[0], 1, ys.shape[-1]))
            xs = xs.repeat(self.model.label_dim, 1, 1)

            assert xs.shape == (
                batch_size * self.model.label_dim,
                x.shape[1],
                x.shape[2],
            )

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -likelihood_func(xs, reconstruction)

        # p(y)
        prior = -torch.squeeze(utils.log_standard_categorical(ys))

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - self.model.kl_weight * self.model.kl_divergence

        L = self.sampler(elbo)

        if is_labelled:
            assert L.shape == (batch_size,)
            return torch.mean(L)

        logits = self.model.classify(x)
        assert xs.shape == (batch_size * self.model.label_dim, x.shape[1], x.shape[2])
        assert L.shape == (batch_size * self.model.label_dim,)
        assert logits.shape == (batch_size, self.model.label_dim)

        L = L.reshape((batch_size, self.model.label_dim))

        # Calculate entropy H(q(y|x)) and sum over all labels
        # H(p) = - integral_x p(x) logp(x) dx
        H = torch.log(logits + 1e-8)
        assert L.shape == logits.shape
        assert H.shape == logits.shape

        H = torch.sum(torch.mul(logits, H), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        assert H.shape == (batch_size,)
        assert L.shape == (batch_size,)

        # Equivalent to -U(x)
        U = L - H

        return torch.mean(U)
