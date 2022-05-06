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
