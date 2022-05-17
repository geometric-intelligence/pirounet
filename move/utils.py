import torch


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

label_features=4

onehot_encoder = make_onehot_encoder(label_features)
y_given=4
y_onehot = onehot_encoder(y_given)
y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
y_title = y_given
import random
y_rand = random.randint(0, label_features-1)
print(y_rand)