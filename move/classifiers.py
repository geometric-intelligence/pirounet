import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import default_config

class LinearClassifier(nn.Module):
    def __init__(self, input_features, h_features_loop, label_features, seq_len, class_neg_slope, class_loops):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(LinearClassifier, self).__init__()
        self.dense = nn.Linear(seq_len*input_features, h_features_loop)
        self.looplayer1 = nn.Linear(h_features_loop, h_features_loop)
        self.looplayer2 = nn.Linear(h_features_loop, h_features_loop)
        self.logits = nn.Linear(h_features_loop, label_features)
        self.class_loops = class_loops
        self.class_neg_slope = class_neg_slope


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1)).float()
        x = F.relu(self.dense(x))
        # for i in range(self.class_loops):
        x = F.leaky_relu(self.looplayer1(x), negative_slope=self.class_neg_slope)
        x = F.leaky_relu(self.looplayer2(x), negative_slope=self.class_neg_slope)
        
        x = F.softmax(self.logits(x), dim=-1)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=default_config.input_features, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TransformerClassifier(PositionalEncoding):
    def __init__(self, input_features, label_features):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(TransformerClassifier, self).__init__()

        self.positional_encoder = PositionalEncoding(dim_model=input_features)
        self.transformer = nn.Transformer(
            d_model=input_features,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dropout=0.1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.01)
        self.logits = nn.Linear(input_features, label_features)

    def forward(self, x, x_lengths, apply_softmax=True):
        x = self.positional_encoder(x)

        # Feed into RNN
        out = self.transformer(x)#shape of out T*N*D

        # Gather the last relevant hidden state
        out = out[-1,:,:] # N*D

        # FC layers
        z = self.dropout(out)
        y_pred = self.logits(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=-1)
        return y_pred
