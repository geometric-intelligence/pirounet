"Utility functions used for training and evaluating the model."

import default_config
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def make_onehot_encoder(label_features):
    """Convert a number to its one-hot representation vector.

    The one-hot representation is also called the 1-of-amount_of_labels.

    Parameters
    ----------
    label_features: int
                    Amount of label categories.

    Returns
    -------
    onehot_encode : function
                    Function that takes in a categorical label integer
                    and outputs its corresponding one-hot.
    """

    def onehot_encode(label):
        y = torch.zeros(label_features)
        if label < label_features:
            y[int(label)] = 1
        return y

    return onehot_encode


def batch_one_hot(y, label_dim):
    """Convert a batch of integer labels to a tensor of one-hot labels.

    Parameters
    ----------
    y:              array
                    Shape = [batch_size, ]
                    Batch of integer labels.
    label_dim:      int
                    Amount of label categories.

    Returns
    -------
    batch_one_hot : tensor
                    Shape = [batch_size, 1, label_dim]
                    Batch of one-hots.
    """
    onehot_encoder = make_onehot_encoder(label_dim)
    batch_one_hot = torch.zeros((1, 1, label_dim))
    for y_i in y:
        y_i_enc = onehot_encoder(y_i)
        y_i_enc = y_i_enc.reshape((1, 1, label_dim))
        batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)

    batch_one_hot = batch_one_hot[1:, :, :]
    return batch_one_hot


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    Uses a uniform prior over categroical distribution.
    The one-hot representation is also called the 1-of-amount_of_labels.

    Parameters
    ----------
    p:          tensor
                Shape = [batch_size, 1, label_features]
                One-hot categorical distribution.

    Returns
    -------
    H(p,u) :    tensor
                Shape = [batch_size, 1]
                Cross entropy for every example in the batch.
    """
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
    because it assigns both all 3 labels (in one-hot encodings)
    to each of the batch_size elements of x

    Parameters
    ----------
    x:      tensor
            Shape = [batch_size, ]
            Batch of integer labels.
    y_dim:  int
            Number of total labels.

    Returns
    -------
    mimic : Variable
            Shape = [batch_size x label_dim, 1, label_dim]
            Tensor assigning every possible label y to each
            element of x.
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.shape[0]
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.to(default_config.device)

    mimic = Variable(generated.float())
    return mimic


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.

    Parameters
    ----------
    tesnor: tensor
            Tensor over which to compute the LSE.
    dim:    int
            Dimension to perform operation over.
    sum_op: operation
            reductive operation to be applied, e.g. torch.sum or torch.mean

    Returns
    -------
    LSE:    tensor
            LogSumExp of input tensor.
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)

    LSE = torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max
    return LSE
