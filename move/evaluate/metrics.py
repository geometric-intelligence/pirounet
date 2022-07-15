"Metrics for quantitative evaluation of a model."

import models.utils as utils
import numpy as np
import torch
from scipy import linalg
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Taken from Action2Motion (Guo et al) at
    https://github.com/EricGuo5513/action-to-motion/

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Parameters
    ----------
    mu1 :       array
                Shape = [h_dim_calssif, ]
                Mean of activation layer from the independently trained classifier given
                generated samples.
    mu2 :       array
                Shape = [h_dim_calssif, ]
                Mean of activation layer from the independently trained classifier given
                validation samples.
    sigma1 :    array
                Shape = [h_dim_calssif, h_dim_calssif]
                Covariance matrix over activations for generated samples.
    sigma2 :    array
                Shape = [h_dim_calssif, h_dim_calssif]
                Covariance matrix over activations for validation samples.

    Returns
    ----------
    fid :       float
                Frechet Inception Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fid


def calculate_activation_statistics(activations):
    """Obtains means and covariance matrices of activation layer.
    Taken from Action2Motion (Guo et al) at
    https://github.com/EricGuo5513/action-to-motion/

    Parameters
    ----------
    activations :   array
                    Shape = [n_seqs , h_dim_classif]
                    Activation layer from the independently trained classifier.

    Returns
    ----------
    mu :            array
                    Shape = [h_dim_classif, ]
                    Mean of activation layer.
    sigma :         array
                    Shape = [h_dim_classif, h_dim_classif]
                    Covariance matrix over activations.
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_diversity_multimodality(activations, labels, label_dim):
    """Obtains diversity and multimodality metrics.

    Metrics are for a given activation layer, thus describing
     the dataset that yielded the activations.

    Taken from Action2Motion (Guo et al) at
    https://github.com/EricGuo5513/action-to-motion/

    Parameters
    ----------
    activations :   array
                    Shape = [n_seqs, h_dim_classif]
                    Activation layer from the (independently trained)
                    classifier fed n_seqs sequences.
    labels :        array
                    Shape = [num_gen_per_lab * label_dim, ]
                    Labels associated to dataset that yielded activations.

    label_dim :     int
                    NUmber of possible categorical labels.

    Returns
    ----------
    diversity :     float
                    Diversity metric.
    multimodality : float
                    Multimodality metric.
    """
    diversity_times = 200
    multimodality_times = 70
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :], activations[second_idx, :])
    diversity /= diversity_times

    multimodality = 0
    label_quotas = np.repeat(multimodality_times, label_dim)
    while np.any(label_quotas > 0):
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not label_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        label_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation, second_activation)

    multimodality /= multimodality_times * label_dim

    return diversity, multimodality


def ajd(model, device, labelled_data, labels, label_dim):
    """Computes the Average Joint Distance (AJD).

    Every input sequence is passed through the model, yielding
    a corresponding reconstruction. The root mean
    square distance is computed between each original and
    reconstructed keypoint, and then averaged over
    all poses in all sequences.

    Parameters
    ----------
    model :         serialized_obj
                    Model to be evaluated.
    device :        torch.device
                    CUDA device identity.
    labelled_data : array
                    Shape = [n_seqs_to_eval, seq_len, input_dim]
                    Sequences to reconstruct.
    labels :        array
                    Shape = [n_seqs_to_eval, label_dim]
                    Labels associated to the sequences meant for
                    recontruction.
    label_dim :     int
                    Amount of possible categorical labels.

    Returns
    ----------
    D :             float
                    Average Joint Distance.
    """

    D_total = 0
    batches_seen = 0

    for i_batch, (x, y) in enumerate(zip(labelled_data, labels)):
        x, y = Variable(x), Variable(y)
        x, y = x.to(device), y.to(device)

        batch_size, seq_len, _ = x.shape

        batch_one_hot = utils.batch_one_hot(y, label_dim)
        y = batch_one_hot.to(device)

        x_recon = model(x, y)

        D_this_batch = 0

        for i in range(len(x)):
            x_seq = x[i]
            x_seq = x_seq.reshape((seq_len, -1, 3)).cpu().data.numpy()
            x_recon_seq = x_recon[i]
            x_recon_seq = x_recon_seq.reshape((seq_len, -1, 3))
            x_recon_seq = x_recon_seq.cpu().data.numpy()

            d = (x_seq - x_recon_seq) ** 2
            d = np.sum(d, axis=2)
            d = np.sqrt(d)
            D = d.reshape(-1)
            D = np.mean(D)
            D_this_batch += D

        batches_seen += 1
        D_total += D_this_batch

    D = D_total / (batches_seen * batch_size)

    return D


def ajd_test(model, device, labelled_data, labels, label_dim):
    """Computes the Average Joint Distance (AJD).

    Made for a small input dataset, i.e. 1 batch or less.

    Parameters
    ----------
    model :         object of the model class torch.nn.Module
                    Model to be evaluated.
    device :        torch.device
                    CUDA device identity.
    labelled_data : array
                    Shape = [n_seqs_to_eval, seq_len, input_dim]
                    Sequences to reconstruct.
    labels :        array
                    Shape = [n_seqs_to_eval, label_dim]
                    Labels associated to the sequences meant for
                    recontruction.
    label_dim :     int
                    Amount of possible categorical labels.

    Returns
    ----------
    D :             float
                    Average Joint Distance.
    """

    D_total = 0
    batches_seen = 0

    x = labelled_data.dataset
    y = labels.dataset

    batch_size, seq_len, _ = x.shape

    batch_one_hot = utils.batch_one_hot(y, label_dim)
    y = batch_one_hot.to(device)

    x_recon = model(torch.tensor(x).to(device), y)

    D_this_batch = 0

    for i in range(len(x)):
        x_seq = x[i]
        x_seq = x_seq.reshape((seq_len, -1, 3))
        x_recon_seq = x_recon[i]
        x_recon_seq = x_recon_seq.reshape((seq_len, -1, 3))
        x_recon_seq = x_recon_seq.cpu().data.numpy()

        d = (x_seq - x_recon_seq) ** 2
        d = np.sum(d, axis=2)
        d = np.sqrt(d)
        D = d.reshape(-1)
        D = np.mean(D)
        D_this_batch += D

        batches_seen += 1
        D_total += D_this_batch

    D = D_total / (batches_seen * batch_size)

    return D


def calc_accuracy(model, device, labelled_data, labels):
    """Calculates the classification accuracy of a model.

    Tests accuracy on given dataset (validation, test, etc.).
    Does not normalize with respsect to amount of data/
    category.

    Parameters
    ----------
    model :         serialized_obj
                    Model to be evaluated.
    device :        torch.device
                    CUDA device identity.
    labelled_data : array
                    Shape = [n_seqs_to_eval, seq_len, input_dim]
                    Sequences to reconstruct.
    labels :        array
                    Shape = [n_seqs_to_eval, label_dim]
                    Labels associated to the sequences meant for
                    recontruction.

    Returns
    ----------
    accuracy :      float
                    Correctly classified examples over entire
                    set of examples (percentage).
    """
    x = torch.from_numpy(labelled_data.dataset)
    y = torch.squeeze(torch.from_numpy(labels.dataset))

    x = x.to(device)

    logits = model.classify(x)

    y_pred = (torch.max(logits, 1).indices).float()

    conf_mat = confusion_matrix(
        y.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
    )

    numerator = conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]
    print(numerator)
    denom = np.concatenate(conf_mat).sum()
    print(denom)
    accuracy = (numerator / denom) * 100
    return accuracy
