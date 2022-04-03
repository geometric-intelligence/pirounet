# Modified from
# https://towardsdatascience.com/
# implementation-differences-in-lstm-layers-tensorflow
# -vs-pytorch-77a31d742f74
import math

import torch
import torch.nn as nn
from torch.nn import init


class LstmEncoder(torch.nn.Module):
    def __init__(self, n_layers, input_features, h_features_loop, latent_dim):
        super().__init__()

        self.n_layers = n_layers
        self.lstm1 = torch.nn.LSTM(
            input_size=input_features, hidden_size=h_features_loop, batch_first=True
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=h_features_loop, batch_first=True
        )
        self.mean_block = torch.nn.Linear(h_features_loop, latent_dim)
        self.logvar_block = torch.nn.Linear(h_features_loop, latent_dim)

    def reparametrize(self, z_mean, z_logvar):
        # # print("reparametrize function called")
        std = torch.exp(0.5 * z_logvar)
        # print("made std")
        eps = torch.randn_like(std)
        # print("# print made eps")
        return eps.mul(std).add_(z_mean)

    def forward(self, inputs):
        """Perform forward pass of the encoder.
        
        Parameters:
        inputs : array-like
            shape=[batch_size, seq_len, input_features]
            where input_features = 3 * n_body_joints.
        """
        # print("starting the forward of encoder."
        # print("the first step is calling layer lstm1")
        # print("input to encoder should have [8,128,159]")
        # print(inputs.shape)
        h1, (h1_T, c1_T) = self.lstm1(inputs)
        # print('h after linear encoder layer')
        # print(h1.shape)
        # print(
        #     "done layer lstm1."
        #     " It returned h1 of shape {} and h1_T of shape{}".format(
        #         h1.shape, h1_T.shape
        #     )
        # )

        # print("Now starting the loop of {}-1 lstm layers".format(self.n_layers))
        for i in range(self.n_layers - 1):
            # print("this is loop iteration {}. Calling layer lstm2".format(i))
            h1, (h1_T, c1_T) = self.lstm2(h1)
            # print(' h and last h of second lstm encoder')
            # print(h1.shape, h1_T.shape)

            # # print(
            #     "done layer lstm 2. "
            #     "lstm2 returns h2 of shape {} and h2_T of shape {}".format(
            #         h2.shape, h2_T.shape
            #     )
            # )

        # print("Now computing the encoder output.")
        # print("calling mean_block")
        h1_T_batchfirst = h1_T.squeeze(axis=0)
        z_mean = self.mean_block(h1_T_batchfirst)
        z_logvar = self.logvar_block(h1_T_batchfirst)
        z_sample = self.reparametrize(z_mean, z_logvar)

        # print("encoder is done")
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
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.linear = torch.nn.Linear(latent_dim, h_features_loop)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=negative_slope)

        self.lstm_loop = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=h_features_loop, batch_first=True
        )

        self.lstm2 = torch.nn.LSTM(
            input_size=h_features_loop, hidden_size=output_features, batch_first=True
        )

    def forward(self, inputs):
        """Perform forward pass of the decoder."""

        h = self.linear(inputs)

        h = self.leakyrelu(h)

        # assert len(h.shape) == 2, h.shape
        h = h.reshape((h.shape[0], 1, h.shape[-1]))  # ,self.seq_len, 1, 1)

        h = h.repeat(1, self.seq_len, 1)
        # print("repeated h:")
        # print(h.shape)
        # print(h[0])
        # h_before = h

        for i in range(self.n_layers - 1):
            h, (h_T, c_T) = self.lstm_loop(h)
            # print(f"h at layer {i}, first batch el, first 10 frmes, first joints in 3D")
            # print(h.shape)
            # print(h[0, :20, :6])   

        h, (h_T, c_T) = self.lstm2(h)
        # print("\n \n h just before output, first batch element, first 10 frames, first joints in 3D:")
        # print(h.shape)
        # print(h[0, :20, :6])

        return h


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of normal distribution parametrised
    by mu and log_var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = (
        -0.5 * math.log(2 * math.pi)
        - log_var / 2
        - (x - mu) ** 2 / (2 * torch.exp(log_var))
    )
    return torch.sum(log_pdf, dim=-1)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


class LstmVAE(torch.nn.Module):
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
        """
        Variational Autoencoder model
        consisting of an (LSTM+encoder)/(decoder+LSTM) pair.
        :param dims: x, z and hidden dimensions of the networks
        """
        super(LstmVAE, self).__init__()

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

        # FIX: This should be in the main.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)  # initialize weight W
                if m.bias is not None:  # initialize b in W*x+b
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.
        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                  = -E[log p(z) - log q(z)]
        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        # -0.5*K.mean(K.sum(1 + auto_log_var -
        # K.square(auto_mean) - K.exp(auto_log_var), axis=-1))

        (mu, log_var) = q_param

        kl = -(torch.sum(1 + log_var - torch.square(mu) - log_var.exp(), axis=-1))

        return kl

    def elbo(self, x_in, x_out, z, q_param, p_param=None):
        # print(
        #     "x_in has shape {}"
        #     " and x_out has shape {}".format(x_in.shape, x_out.shape)
        # )
        recon_loss = (x_in - x_out) ** 2 # 8, 128, 159
        print("\n\n recon loss")
        print(recon_loss.shape)
        print(recon_loss[0])
        recon_loss = torch.sum(recon_loss, axis=2)
        print(recon_loss.shape)
        print(recon_loss[0])
        recon_loss = torch.sum(recon_loss)
        # x_out is sequence
        # 0.5*K.mean(K.sum(K.square(auto_input - auto_output), axis=-1))

        regul_loss = self._kld(z, q_param, p_param)
        # -0.5*K.mean(K.sum(
        # 1 + auto_log_var - K.square(auto_mean) - K.exp(auto_log_var), axis=-1))

        return recon_loss + self.kl_weight * regul_loss

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data, has shape=[batch_size, seqlen, input_features].
        :return: reconstructed input
        """
        z_sample, z_mean, z_log_var = self.encoder(x)

        q_param = (z_mean, z_log_var)

        # FIX: KLD was not called correctly
        self.kl_divergence = self._kld(z_sample, q_param)

        x_mean = self.decoder(z_sample)

        return x_mean, z_sample, z_mean, z_log_var


# def setup_gpus():
#     # use tensorflow backend
#     # set random seeds
#     # tf.set_random_seed(1)
#     # np.random.seed(1)
#     # identify available GPU's
#     #     gpus = K.tensorflow_backend.
# _get_available_gpus() # works with TF 1 (?)
#     #     gpus = tf.config.experimental.list_physical_devices('GPU') # works with TF 2

#     os.environ[
#         "CUDA_VISIBLE_DEVICES"
#     ] = "3"  # pick a number < 4 on ML4HEP; < 3 on Voltan
#     gpu_options = tf.GPUOptions(
# allow_growth=True, per_process_gpu_memory_fraction=0.5)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     # allow dynamic GPU memory allocation
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.compat.v1.Session(config=config)
#     #     # print("GPUs found: {}".format(len(gpus)))
#     return ()

