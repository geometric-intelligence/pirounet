import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, input_features, h_features_loop, label_features, seq_len):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(LinearClassifier, self).__init__()
        self.dense = nn.Linear(seq_len*input_features, h_features_loop)
        self.logits = nn.Linear(h_features_loop, label_features)


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1)).float()
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)

        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_features, h_features_loop, label_features, seq_len):
        """
        Transformer classifier
        with softmax output.
        """
        super(TransformerClassifier, self).__init__()
        self.dense = nn.Linear(seq_len*input_features, h_features_loop)
        self.logits = nn.Linear(h_features_loop, label_features)


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1)).float()
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)

        return x