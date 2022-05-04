"""Architectures of neural networks.

The LstmVAE has been adapted from beyond-imitation, using:
https://towardsdatascience.com/
implementation-differences-in-lstm-layers-tensorflow
-vs-pytorch-77a31d742f74
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class LstmEncoder(torch.nn.Module):
    """Encoder with LSTM layers."""

    def __init__(
        self, n_layers, input_features, h_features_loop, latent_dim, label_features=None
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_features = input_features
        self.h_features_loop = h_features_loop
        self.latent_dim = latent_dim
        self.label_features = label_features

        if label_features:
            total_input_features = input_features + label_features
        else:
            total_input_features = input_features

        self.lstm1 = torch.nn.LSTM(
            input_size=total_input_features,
            hidden_size=h_features_loop,
            batch_first=True,
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=h_features_loop, batch_first=True
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

        h, (h_last_t, c_last_t) = self.lstm1(inputs)
        logging.debug(
            f"LSTM1 gives h of shape {h.shape} & h_last_t of shape {h_last_t.shape} "
        )

        for i in range(self.n_layers - 1):
            logging.debug(f"LSTM2 loop iteration {i}/{self.n_layers-1}.")
            h, (h_last_t, c_last_t) = self.lstm2(h, (h_last_t, c_last_t))
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


class LstmDecoder(torch.nn.Module):
    """Decoder with LSTM layers."""

    def __init__(
        self,
        n_layers,
        output_features,
        h_features_loop,
        latent_dim,
        seq_len,
        negative_slope,
        label_features,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.output_features = output_features
        self.h_features_loop = h_features_loop
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.label_features = label_features

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=negative_slope)

        if label_features:
            input_features_decoder = h_features_loop + label_features
            total_latent_dim = latent_dim + label_features
        else:
            input_features_decoder = h_features_loop
            total_latent_dim = latent_dim

        print(f"input_features_decoder: {input_features_decoder}")

        self.linear = torch.nn.Linear(total_latent_dim, h_features_loop)

        self.lstm_loop = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=h_features_loop, batch_first=True
        )

        self.lstm2 = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=output_features, batch_first=True
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
        batch_size, _ = inputs.shape
        logging.debug(f"- Decoder inputs are of shape {inputs.shape}")
        print("inputs before going into h")
        print(inputs.shape)

        h = self.linear(inputs)
        h = self.leakyrelu(h)

        assert h.shape == (batch_size, self.h_features_loop)
        h = h.reshape((h.shape[0], 1, h.shape[-1]))
        h = h.repeat(1, self.seq_len, 1)
        assert h.shape == (batch_size, self.seq_len, self.h_features_loop)
        print("shape of h")
        print(h.shape)

        # input_zeros = torch.zeros_like(h)
        # h0 = h[:, 0, :]
        # h0 = h0.reshape((1, h0.shape[0], h0.shape[-1]))
        # c0 = h0

        h, (h_last_t, c_last_t) = self.lstm_loop(h)

        for i in range(self.n_layers - 1):
            logging.debug(f"LSTM loop iteration {i}/{self.n_layers-1}.")
            h, (h_last_t, c_last_t) = self.lstm_loop(h)
            assert h.shape == (batch_size, self.seq_len, self.h_features_loop)
            logging.debug(f"First batch example, first 20t: {h[0, :20, :4]}")

        h, _ = self.lstm2(h)
        assert h.shape == (batch_size, self.seq_len, self.output_features)
        logging.debug(f"1st batch example, 1st 20t, 1st 2 joints: {h[0, :20, :6]}")
        logging.debug(f"1st batch example, last 20t, 1st 2 joints: {h[0, :-20, :6]}")
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
        h_features_loop=32,
        latent_dim=32,
        kl_weight=0,
        output_features=3 * 53,
        seq_len=128,
        negative_slope=0.2,
        with_rotation_layer=True,
        label_features=None,
    ):
        super(LstmVAE, self).__init__()
        self.latent_dim = latent_dim
        self.with_rotation_layer = with_rotation_layer

        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_features=input_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            label_features=label_features,
        )
        self.decoder = LstmDecoder(
            n_layers=n_layers,
            output_features=output_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            seq_len=seq_len,
            negative_slope=negative_slope,
            label_features=label_features,
        )
        self.kl_divergence = 0
        self.kl_weight = kl_weight

        assert input_features == output_features

        # This initializes the weights and biases
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
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


class Classifier(nn.Module):
    def __init__(self, input_features, h_features_loop, label_features, seq_len):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        self.dense = nn.Linear(seq_len*input_features, h_features_loop)
        self.logits = nn.Linear(h_features_loop, label_features)

    
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.dense(x.reshape((batch_size, -1))))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(LstmVAE):
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

        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_features=input_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            label_features=label_features,
        )

        self.decoder = LstmDecoder(
            n_layers=n_layers,
            output_features=output_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            seq_len=seq_len,
            negative_slope=negative_slope,
            label_features=label_features,
        )

        self.classifier = Classifier(input_features, h_features_loop, label_features, seq_len)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
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
        print("y before repeat")
        print(y.shape)
        y_for_encoder = y.repeat((1, self.seq_len, 1))

        print('SHAPE OF x')
        print(x.shape)
        print('SHAPE OF Y')
        print(y.shape)

        z, z_mu, z_log_var = self.encoder(torch.cat([x, y_for_encoder], dim=2).float())

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        print("z")
        print(z.shape)
        print(y.shape)

        y_for_decoder = y.reshape((y.shape[0], y.shape[-1]))
        print("y for dec")
        print(y_for_decoder.shape)
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
            ys = enumerate_discrete(xs, self.model.label_features)
            xs = xs.repeat(self.model.label_features, 1, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -likelihood_func(xs, reconstruction)
        print("SHAPE OF LIKELIHOOD")
        print(likelihood.shape)

        # p(y)
        prior = -log_standard_categorical(ys)
        print("SHAPE OF PRIOR")
        print(prior.shape)
        print("kl divergence")
        print(self.model.kl_divergence.shape)
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
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

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
