import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_features, h_features_loop, label_features):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        self.dense = nn.Linear(input_features, h_features_loop)
        self.logits = nn.Linear(h_features_loop, label_features)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x
