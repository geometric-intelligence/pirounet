"""Architectures of DGM LSTM VAE."""

import models.utils as utils
import torch.nn
import torch.nn.functional as F
from models.classifiers import ActorClassifier
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

        # self.classifier = LinearClassifier(
        #     input_dim=input_dim,
        #     h_dim=h_dim_classif,
        #     label_dim=label_dim,
        #     seq_len=seq_len,
        #     neg_slope=neg_slope_classif,
        #     n_layers=n_layers_classif,
        # )

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
            Shape=[batch_size, seq_len, input_dim].

        y : array-like
            one hot encoder.
            Shape=[batch_size, label_dim].

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

    def reconstruction_loss(x, x_recon):
        assert x.ndim == x_recon.ndim == 3
        batch_size, seq_len, _ = x.shape
        recon_loss = (x - x_recon) ** 2

        recon_loss = torch.mean(recon_loss, axis=(1, 2))

        assert recon_loss.shape == (batch_size,)
        return recon_loss

    def forward(self, x, y=None, likelihood_func=reconstruction_loss):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = utils.enumerate_discrete(xs, self.model.label_dim)
            ys = ys.reshape((ys.shape[0], 1, ys.shape[-1]))
            xs = xs.repeat(self.model.label_dim, 1, 1)

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
