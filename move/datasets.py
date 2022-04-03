import logging
import os
from glob import glob

import numpy as np
import torch


def load_mariel_raw(pattern="data/mariel_*.npy"):
    """Load six datasets and perform minimal preprocessing.

    Processing amunts to center each dancer, such that
    the barycenter becomes 0.
    """
    datasets = {}
    ds_all = []

    exclude_points = [26, 53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0

    for f in sorted(glob(pattern)):
        ds_name = os.path.basename(f)[7:-4]
        # print("loading:", ds_name)
        ds = np.load(f).transpose((1, 0, 2))
        ds = ds[500:-500, point_mask]
        # print("\t Shape:", ds.shape)

        ds[:, :, 2] *= -1
        # print("\t Min:", np.min(ds, axis=(0, 1)))
        # print("\t Max:", np.max(ds, axis=(0, 1)))

        # ds = filter_points(ds)

        datasets[ds_name] = ds
        ds_all.append(ds)

    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1])

    ds_all = np.concatenate(ds_all)
    # print("Full data shape:", ds_all.shape)
    # # print("Offsets:", ds_offsets)

    # # print(ds_all.min(axis=(0,1)))
    low, hi = np.quantile(ds_all, [0.01, 0.99], axis=(0, 1))
    xy_min = min(low[:2])
    xy_max = max(hi[:2])
    xy_range = xy_max - xy_min
    ds_all[:, :, :2] -= xy_min
    ds_all *= 2 / xy_range
    ds_all[:, :, :2] -= 1.0

    # it's also useful to have these datasets centered, i.e. with the x and y offsets
    # subtracted from each individual frame

    ds_all_centered = ds_all.copy()
    ds_all_centered[:, :, :2] -= ds_all_centered[:, :, :2].mean(axis=1, keepdims=True)

    datasets_centered = {}
    for ds in datasets:
        datasets[ds][:, :, :2] -= xy_min
        datasets[ds] *= 2 / xy_range
        datasets[ds][:, :, :2] -= 1.0
        datasets_centered[ds] = datasets[ds].copy()
        datasets_centered[ds][:, :, :2] -= datasets[ds][:, :, :2].mean(
            axis=1, keepdims=True
        )

    # # print(ds_all.min(axis=(0,1)))
    low, hi = np.quantile(ds_all, [0.01, 0.99], axis=(0, 1))
    return ds_all, ds_all_centered, datasets, datasets_centered, ds_counts


def get_mariel_data(config):
    ds_all, ds_all_centered, _, _, _ = load_mariel_raw()
    my_data = ds_all_centered.reshape((ds_all.shape[0], -1))

    logging.info("Make seq_data of shape [number of seq, seq_len, input_features]")
    seq_data = np.zeros(
        (my_data.shape[0] - config.seq_len, config.seq_len, my_data.shape[1])
    )
    for i in range((ds_all.shape[0] - config.seq_len)):
        seq_data[i] = my_data[i : i + config.seq_len]

    logging.info("Make training data and validing data")
    five_perc = int(round(seq_data.shape[0] * 0.05))
    ninety_perc = seq_data.shape[0] - (2 * five_perc)
    valid_ds = seq_data[:five_perc, :, :]
    test_ds = seq_data[five_perc : (2 * five_perc), :, :]
    train_ds = seq_data[ninety_perc:, :, :]

    logging.info("Make torch tensor in batches")
    data_train_torch = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size
    )
    data_valid_torch = torch.utils.data.DataLoader(
        valid_ds, batch_size=config.batch_size
    )
    data_test_torch = torch.utils.data.DataLoader(test_ds, batch_size=1)
    return data_train_torch, data_valid_torch, data_test_torch
