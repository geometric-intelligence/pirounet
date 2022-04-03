"""Architectures of neural networks.

The LstmVAE has been adapted from beyond-imitation, using:
https://towardsdatascience.com/
implementation-differences-in-lstm-layers-tensorflow
-vs-pytorch-77a31d742f74
"""

import logging

import torch


class LstmEncoder(torch.nn.Module):
    def __init__(self, n_layers, input_features, h_features_loop, latent_dim):
        super().__init__()
        self.n_layers = n_layers
        self.input_features = input_features
        self.h_features_loop = h_features_loop
        self.latent_dim = latent_dim

        self.lstm1 = torch.nn.LSTM(
            input_size=input_features, hidden_size=h_features_loop, batch_first=True
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
        assert inputs.ndim == 3, inputs.shape
        assert inputs.shape[-1] == self.input_features
        batch_size, seq_len, _ = inputs.shape

        logging.debug(f"Encoder inputs of shape {inputs.shape}")
        h, (h_last_t, _) = self.lstm1(inputs)
        logging.debug(f"LSTM1 gives h of shape: {h.shape}")
        logging.debug(f"LSTM1 gives h_last_t of shape {h_last_t.shape}")

        for i in range(self.n_layers - 1):
            logging.debug(f"- # Encoder LSTM loop iteration {i}/{self.n_layers-1}.")
            h, (h_last_t, _) = self.lstm2(h)
            assert h.shape == (batch_size, seq_len, self.h_features_loop)
            assert h_last_t.shape == (batch_size, 1, self.h_features_loop)

        logging.debug("Computing the encoder output.")
        h1_last_t = h_last_t.squeeze(axis=0)
        assert h1_last_t.shape == (batch_size, self.h_features_loop)
        z_mean = self.mean_block(h1_last_t)
        z_logvar = self.logvar_block(h1_last_t)
        z_sample = self.reparametrize(z_mean, z_logvar)

        logging.debug("Encoder is done.")
        return z_sample, z_mean, z_logvar


class LstmDecoder(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        output_features,
        h_features_loop,
        latent_dim,
        seq_len,
        negative_slope,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.output_features
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.linear = torch.nn.Linear(latent_dim, h_features_loop)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=negative_slope)

        self.lstm_loop = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=h_features_loop, batch_first=True
        )

        self.lstm2 = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=output_features, batch_first=True
        )

    def forward(self, inputs):
        """Perform forward pass of the decoder.

        Parameters
        ----------
        inputs : array-like, shape=[batch_size, latent_dim]
        """
        assert inputs.ndim == 2
        assert inputs.shape[-1] == self.latent_dim
        batch_size, _ = inputs.shape

        h = self.linear(inputs)
        h = self.leakyrelu(h)

        assert h.shape == (batch_size, self.h_features_loop)
        h = h.reshape((h.shape[0], 1, h.shape[-1]))
        h = h.repeat(1, self.seq_len, 1)
        assert h.shape == (batch_size, self.seq_len, self.h_features_loop)

        for i in range(self.n_layers - 1):
            logging.debug(f"- # Decoder LSTM loop iteration {i}/{self.n_layers-1}.")
            h, _ = self.lstm_loop(h)
            assert h.shape == (batch_size, self.seq_len, self.h_features_loop)
            logging.debug(f"First batch example, first 20t: {h[0, :20, :4]}")

        h, _ = self.lstm2(h)
        assert h.shape == (batch_size, self.seq_len, self.output_features)
        logging.debug(f"1st batch example, 1st 20t, 1st 2 joints: {h[0, :20, :6]}")
        logging.debug(f"1st batch example, kast 20t, first 2 joints: {h[0, :-20, :6]}")
        return h


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
    ):
        super(LstmVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = LstmEncoder(
            n_layers=n_layers,
            input_features=input_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
        )
        self.decoder = LstmDecoder(
            n_layers=n_layers,
            output_features=output_features,
            h_features_loop=h_features_loop,
            latent_dim=latent_dim,
            seq_len=seq_len,
            negative_slope=negative_slope,
        )
        self.kl_divergence = 0
        self.kl_weight = kl_weight

        assert input_features == output_features

        # This initialize the weights and biases
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

        Formula in Keras was (reconstrution loss):
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
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        Parameters
        ----------
        x : array-like
            input data.
            Shape=[batch_size, seqlen, input_features].

        Returns
        -------
        x_mean : array-like
            reconstructed input
        """
        z, z_mean, z_log_var = self.encoder(x)

        q_param = (z_mean, z_log_var)

        self.kl_divergence = self._kld(z, q_param)

        x_mean = self.decoder(z)
        return x_mean, z, z_mean, z_log_var
