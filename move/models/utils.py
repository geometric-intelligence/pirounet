import torch


def make_onehot_encoder(label_features):
    """Convert a number to its one-hot representation vector.

    The one-hot representation is also called the 1-of-amount_of_labels.

    Parameters
    ----------
    label_features: int
        Length of vector

    Returns
    -------
    onehot_encode : function
    """

    def onehot_encode(label):
        y = torch.zeros(label_features)
        if label < label_features:
            y[int(label)] = 1
        return y

    return onehot_encode


label_features = 4

onehot_encoder = make_onehot_encoder(label_features)
y_given = 4
y_onehot = onehot_encoder(y_given)
y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
y_title = y_given
import random

y_rand = random.randint(0, label_features - 1)
print(y_rand)
