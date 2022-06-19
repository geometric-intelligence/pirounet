import numpy as np
from scipy import linalg
import torch
from torch.autograd import Variable

import models.utils as utils

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Taken from Action2Motion (Guo et al) at
    https://github.com/EricGuo5513/action-to-motion/

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_diversity_multimodality(activations, labels, num_labels):
    print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 70
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    print('=== Evaluating Multimodality ===')
    multimodality = 0
    labal_quotas = np.repeat(multimodality_times, num_labels)
    while np.any(labal_quotas > 0):
        # print(labal_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not labal_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        labal_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)

    multimodality /= (multimodality_times * num_labels)

    return diversity, multimodality

def ajd(model, device, labelled_data_valid, labels_valid, label_dim):
    """
    Computes the Average Joint Distance (AJD) over validation data
    for a given model.

    """

    D_total = 0
    batches_seen = 0

    for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):
        x, y = Variable(x), Variable(y)
        x, y = x.to(device), y.to(device)

        batch_size, seq_len, _ = x.shape

        batch_one_hot = utils.batch_one_hot(y, label_dim)
        y = batch_one_hot.to(device)

        x_recon = model(x, y)  # has shape [batch_size, seq_len, 159]
        # sum over all 159 coordinates and 

        D_this_batch = 0

        for i in range(len(x)):
            x_seq = x[i]  #pick one sequence in batch
            x_seq = x_seq.reshape((seq_len, -1, 3)).cpu().data.numpy()
            x_recon_seq = x_recon[i]
            x_recon_seq = x_recon_seq.reshape((seq_len, -1, 3))
            x_recon_seq= x_recon_seq.cpu().data.numpy()

            d = (x_seq - x_recon_seq)**2
            d = np.sum(d, axis = 2)
            d = np.sqrt(d) # shape [40,53]
            D = d.reshape(-1)
            D = np.mean(D)
            D_this_batch += D # make D for all sequences in batch, [batchsize,]

        #D for all batches
        batches_seen += 1
        D_total += D_this_batch

    # average D for valid dataset
    D_valid = D_total/ (batches_seen * batch_size)

    return D_valid