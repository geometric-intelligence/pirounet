"""Architectures of DGM LSTM VAE."""

import torch.nn
import torch.nn.functional as F
from models.classifiers import LinearClassifier
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

        self.classifier = LinearClassifier(
            input_dim=input_dim,
            h_dim=h_dim_classif,
            label_dim=label_dim,
            seq_len=seq_len,
            neg_slope=neg_slope_classif,
            n_layers=n_layers_classif,
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
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
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
            ys = enumerate_discrete(xs, self.model.label_dim)
            ys = ys.reshape((ys.shape[0], 1, ys.shape[-1]))
            xs = xs.repeat(self.model.label_dim, 1, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        # print('CHECKING IF SAMPLED X AND Y HAVE NANS')
        if torch.isnan(xs).any():
            print("sampled xs has nan")
        if torch.isnan(ys).any():
            print("sampled ys has nan")
        reconstruction = self.model(xs, ys)
        # print('CHECKING IF RECONSTRUCTION HAS NAN')
        if torch.isnan(reconstruction).any():
            print("reconstruction has nan")

        # p(x|y,z)
        likelihood = -likelihood_func(xs, reconstruction)
        # print('CHECKING IF LIKELIHOOD HAS NAN')
        if torch.isnan(likelihood).any():
            print("likelihood has nan")

        # p(y)
        prior = -torch.squeeze(log_standard_categorical(ys))
        # print('CHECKING IF PRIOR HAS NAN')
        if torch.isnan(prior).any():
            print("prior has nan")

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
    :param p: one-hot categorical distribution with shape [batch_size, 1, label_dim]
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
