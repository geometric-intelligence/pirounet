import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import PositionalEncoding

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

        logits = F.softmax(self.layers[-1](x), dim=1)
        return logits, 0

class FID_LinearClassifier(nn.Module):
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
        super(FID_LinearClassifier, self).__init__()
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

        for layer in self.layers[:(self.n_layers - 1)]:

            x = F.leaky_relu(layer(x), negative_slope=self.neg_slope) 

        activation = x #[batch_size, h_dim_class]
        logits = F.softmax(self.layers[-1](x), dim=1)

        return logits, activation


class FID_lstm_Classifier(nn.Module):
    def __init__(
        self,
        input_dim,
        h_dim,
        h_dim_class,
        label_dim,
        n_layers,
        n_layers_class,
        return_activation
    ):
        """
        Model that runs classifier for FID, diversity, and multimodul diversity
        """
        super(FID_lstm_Classifier, self).__init__()
        
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.label_dim = label_dim
        self.n_layers_class = n_layers_class
        self.return_activation = return_activation

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=h_dim,
            batch_first=True,
            num_layers=n_layers,
        )

        self.dense = nn.Linear(h_dim, h_dim_class)

        self.lin_layers = nn.ModuleList()
        for i in range(int(n_layers_class)):
            self.lin_layers.append(nn.Linear(h_dim_class, h_dim_class))

        self.lin_layers.append(nn.Linear(h_dim_class, label_dim))

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        """Classify input x into logits.

        Parameters
        ----------
        inputs : array-like
            Shape=[batch_size, seq_len, input_dim]

        Returns
        -------
        logits : array-like
                 Shape=[batch_size, label_dim]
        """

        batch_size, seq_len, _ = inputs.shape

        h, (h_last_t_all, c_last_t) = self.lstm(inputs.float())

        assert h.shape == (batch_size, seq_len, self.h_dim)
        assert h_last_t_all.shape == (self.n_layers, batch_size, self.h_dim)
        h_last_t = h_last_t_all[self.n_layers - 1, :, :]
        assert h_last_t.shape == (batch_size, self.h_dim)

        x = self.dense(h_last_t)
        for layer in self.lin_layers[:(self.n_layers_class - 1)]:
            x = F.leaky_relu(layer(x), negative_slope=0.01) 

        logits = F.softmax(self.lin_layers[-1](x), dim=1)
        # print(self.dense.weight.grad)
        if not self.return_activation:
            return logits
        if self.return_activation:
            return logits

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
        """This is Encoder_TRANSFORMER from Actor, turned into a classifier"""
        super().__init__()
        self.seq_len = seq_len
        self.label_dim = label_dim
        self.input_dim = input_dim
        self.embed_dim = embed_dim

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

        self.dense = nn.Linear(embed_dim, label_dim)

    def forward(self, x):
        """Perform forward pass of the linear classifier.

        Parameters
        ----------
        x : array-like, shape=[batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim

        # Transforming to shape seq_len, batch_size, input_dim
        # to match Actor's architecture
        x = x.permute((1, 0, 2))
        assert x.shape == (self.seq_len, batch_size, self.input_dim)

        # embedding of the skeleton
        x = self.embedding(x.float())
        assert x.shape == (self.seq_len, batch_size, self.embed_dim)

        # add positional encoding
        xseq = self.seq_positional_encoding(x)

        final = self.seq_transformer_encoder(xseq)
        assert final.shape == (seq_len, batch_size, self.embed_dim)
        mu = final[0]

        # ignoring logvar: only here for compatibility with Actor
        logvar = final[1]
        assert mu.shape == (batch_size, self.embed_dim)
        assert logvar.shape == (batch_size, self.embed_dim)

        y_pred = self.dense(mu)
        assert y_pred.shape == (batch_size, self.label_dim)

        logits = F.softmax(y_pred, dim=-1)

        return logits
