import default_config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        h_dim,
        label_dim,
        seq_len,
        neg_slope,
        n_layers,
    ):
        """
        Multi hidden layer classifier
        with softmax output.
        Can be ReLu or LeakyRelu
        activated.
        """
        super(LinearClassifier, self).__init__()
        self.dense = nn.Linear(seq_len * input_dim, h_dim)

        self.n_layers = n_layers
        self.neg_slope = neg_slope

        self.layers = nn.ModuleList()
        for i in range(int(n_layers)):
            self.layers.append(nn.Linear(h_dim, h_dim))

        self.layers.append(nn.Linear(h_dim, label_dim))

    def forward(self, x):
        """Perform forward pass of the linear classifier.

        Parameters
        ----------
        x : array-like, shape=[batch_size, seq_len, input_dim]
        """

        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1)).float()
        x = F.relu(self.dense(x))

        if self.neg_slope is not None and not 0:
            for layer in self.layers[:-1]:
                x = F.leaky_relu(layer(x), negative_slope=self.neg_slope)

        else:
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))

        logits = F.softmax(self.layers[-1](x))
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # d_model
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TransformerClassifier(PositionalEncoding):
    def __init__(self, input_dim, label_dim):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(TransformerClassifier, self).__init__()

        self.positional_encoder = PositionalEncoding(dim_model=input_dim)
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dropout=0.1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.01)
        self.logits = nn.Linear(input_dim, label_dim)

    def forward(self, x, x_lengths, apply_softmax=True):
        x = self.positional_encoder(x)

        # Feed into RNN
        out = self.transformer(x)  # shape of out T*N*D

        # Gather the last relevant hidden state
        out = out[-1, :, :]  # N*D

        # FC layers
        z = self.dropout(out)
        y_pred = self.logits(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=-1)
        return y_pred


class ActorClassifier(torch.nn.Module):
    def __init__(
        self,
        seq_len,
        label_dim,
        input_dim,
        embed_dim=256,
        dim_feedforward=1024,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        activation="gelu",
        **kargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_dim = label_dim
        self.input_dim = input_dim

        self.mu_query = nn.Parameter(torch.randn(self.label_dim, embed_dim))
        self.sigma_query = nn.Parameter(torch.randn(self.label_dim, embed_dim))

        self.embedding = nn.Linear(self.input_dim, embed_dim)
        self.seq_positional_encoding = PositionalEncoding(embed_dim, dropout)

        seq_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.seq_transformer_encoder = nn.TransformerEncoder(
            seq_transformer_encoder_layer, num_layers=n_layers
        )

        self.logits = nn.Linear(embed_dim, label_dim)

    def forward(self, x):
        """Perform forward pass of the linear classifier.

        Parameters
        ----------
        x : array-like, shape=[batch_size, seq_len, input_dim]
        """
        _, seq_len, input_dim = x.shape
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim

        # Transforming to shape seq_len, batch_size, input_dim
        # to match Actor's architecture
        x = x.permute((1, 0, 2))
        seq_len, _, input_dim = x.shape
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim

        # embedding of the skeleton
        x = self.embedding(x.float())

        # adding the mu and sigma queries
        # xseq = torch.cat((self.mu_query[y][None], self.sigma_query[y][None], x), axis=0)

        # add positional encoding
        xseq = self.seq_positional_encoding(x)

        final = self.seq_transformer_encoder(xseq)
        mu = final[0]

        # ignoring logvar: only here for compatibility with Actor
        logvar = final[1]

        y_pred = self.logits(mu)

        y_pred = F.softmax(y_pred, dim=-1)

        return y_pred
