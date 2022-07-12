"Defines classifier (for training DGM and indepndent training)."

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

        Parameters
        ----------
        input_dim : int
                    Number of input features in pose
                    (keypoints * 3 dimensions).
        h_dim : int
                Number of nodes in hidden layers.
        label_dim : int
                    Amount of categorical labels.
        seq_len :   int
                    Number of poses in a sequence.
        neg_slope : int
                    Slope for LeakkyRelu activation.
        n_layers :  int
                    Number of linear layers.
             
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
        x : array
            Shape = [batch_size, seq_len, input_dim]
            Input batch of sequences.
        
        Returns
        ----------
        logits :    array
                    Shape = [batch_size, label_dim]
                    Batch of normalized vectors representing
                    probability of each categorical label
                    for each sequence.
        activation :    array
                        Shape = [batch_size, h_dim_classif]
                        Output of before-last hidden layer
        """

        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1)).float()
        x = F.relu(self.dense(x))

        if self.neg_slope is not None and not 0:
            for layer in self.layers[:-1]:
                x = F.leaky_relu(layer(x), negative_slope=self.neg_slope)
            activation = x

        else:
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            activation = x 

        logits = F.softmax(self.layers[-1](x), dim=1)
        return logits, activation