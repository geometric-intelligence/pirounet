import torch
import torch.nn.functional as F
from torch.autograd import Variable


def make_onehot_encoder(label_features):
    """
    Converts a number to its one-hot or 1-of-amount_of_labels representation
    vector.
    :param amount_of_labels: (int) length of vector
    :return: onehot function
    """
    def onehot_encode(label):
        y = torch.zeros(label_features)
        if label < label_features:
            y[int(label)] = 1
        return y
    return onehot_encode


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
