"""Utils to process raw data and make them into tensors."""

import logging
import os
from glob import glob

import numpy as np
import torch
import pickle


def augment_by_rotations(seq_data, augmentation_factor):
    """Augment the dataset of sequences.

    For each input sequence, generate N new sequences, where
    each new sequence is obtained by rotating the input sequence
    with a different rotation angle.

    Notes
    -----
    - N is equal to augmentation_factor.
    - Only rotations of axis z are considered.

    Parameters
    ----------
    seq_data : array-like
        Original dataset of sequences.
        Shape=[n_seqs, seq_len, input_features]
    augmentation_factor : int
        Multiplication factor, i.e. how many new sequences
        are generated from one given sequence.

    Returns
    -------
    rotated_seq_data : array-like
        Augmented dataset of sequences.
        Shape=[augmentation_factor*n_seqs, seq_len, input_features]
    """
    rotated_seq_data = []
    for seq in seq_data:
        thetas = np.random.uniform(0, 2 * np.pi, size=(augmentation_factor,))
        c_thetas = np.cos(thetas)
        s_thetas = np.sin(thetas)
        rotation_mats = [
            np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for c, s in zip(c_thetas, s_thetas)
        ]
        rotation_mats = np.stack(rotation_mats, axis=0)
        assert rotation_mats.shape == (augmentation_factor, 3, 3), rotation_mats.shape
        seq_len, input_features = seq.shape
        seq = seq.reshape((seq_len, -1, 3))
        rotated_seq = np.einsum("...j, nij->...ni", seq, rotation_mats)
        rotated_seq = rotated_seq.reshape(
            (augmentation_factor, seq_len, input_features)
        )
        rotated_seq_data.append(rotated_seq)

    seq_data = np.stack(rotated_seq_data, axis=0).reshape((-1, seq_len, input_features))
    return seq_data


def load_mariel_raw(pattern="data/mariel_*.npy"):
    """Load six datasets and perform minimal preprocessing.

    Processing amunts to center each dancer, such that
    the barycenter becomes 0.

    From Pettee 2019:
    Each frame of the dataset is transformed such that the
    overall average (x,y) position per frame is centered at
    the same point and scaled such that all of the coordinates
    fit within the unit cube.
    """
    datasets = {}
    ds_all = []

    exclude_points = [26, 53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0

    logging.info("Loading raw datasets:")
    for f in sorted(glob(pattern)):
        ds_name = os.path.basename(f)[7:-4]
        ds = np.load(f).transpose((1, 0, 2))
        ds = ds[500:-500, point_mask]
        logging.info(f"- {f} of shape {ds.shape}")

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

    # it's also useful to have these datasets centered,
    # i.e. with the x and y offsets
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

def load_labels(filepath="/home/papillon/move/move/data/labels.csv"):

    file = open(filepath)
    labels_with_index = np.loadtxt(file, delimiter=",")

    labels = np.delete(labels_with_index, 0, axis=1)
    last_label_index = int(labels_with_index[-1][0])
    labelled_seq_len =  last_label_index - int(labels_with_index[-2][0])
    
    return labels, last_label_index, labelled_seq_len


def load_aist_raw(pattern="data/aist/*.pkl"):
    """Load six datasets and perform minimal preprocessing.

    Processing amunts to center each dancer, such that
    the barycenter becomes 0.
    """
    datasets = {}
    ds_all = []

    exclude_points = [26, 53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0

    logging.info("Loading raw datasets:")
    for f in sorted(glob(pattern)):
        ds_name = os.path.basename(f)[7:-4]
        #WITH PICKLE
        data = pickle.load(open(f, "rb"))

        kp3d = data["keypoints3d"]
        kp3d_optim = data["keypoints3d_optim"]
        #ds = ds[500:-500, point_mask]
        logging.info(f"- {f} of shape {kp3d.shape}")

        datasets[ds_name] = kp3d
        ds_all.append(kp3d)

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

    # it's also useful to have these datasets centered,
    # i.e. with the x and y offsets
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


def get_mariel_data(config, augmentation_factor=1):
    """Transform mariel data into train/val/test torch loaders.

    Note: Pettee 2019 keeps augmentation_factor=1.
    """
    ds_all, ds_all_centered, _, _, _ = load_mariel_raw()
    my_data = ds_all_centered.reshape((ds_all.shape[0], -1))

    seq_data = np.zeros(
        (my_data.shape[0] - config.seq_len, config.seq_len, my_data.shape[1])
    )
    for i in range((ds_all.shape[0] - config.seq_len)):
        seq_data[i] = my_data[i : i + config.seq_len]
    logging.info(f"Preprocessing: Load seq_data of shape {seq_data.shape}")

    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Augmented seq_data has shape: {seq_data.shape}")

    five_perc = int(round(seq_data.shape[0] * 0.05))
    ninety_perc = seq_data.shape[0] - (2 * five_perc)
    train_ds = seq_data[:ninety_perc, :, :]
    valid_ds = seq_data[ninety_perc : (ninety_perc + five_perc), :, :]
    test_ds = seq_data[(ninety_perc + five_perc) :, :, :]
    logging.info(f">> Train ds has shape {train_ds.shape}")
    logging.info(f">> Valid ds has shape {valid_ds.shape}")
    logging.info(f">> Test ds has shape {test_ds.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")
    data_train_torch = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size
    )
    data_valid_torch = torch.utils.data.DataLoader(
        valid_ds, batch_size=config.batch_size
    )
    data_test_torch = torch.utils.data.DataLoader(test_ds, batch_size=1)
    return data_train_torch, data_valid_torch, data_test_torch


def get_aist_data(config, augmentation_factor=1):
    """Transform AIST++ data into train/val/test torch loaders.
    """

    ds_all, ds_all_centered, _, _, _ = load_aist_raw()
    my_data = ds_all_centered.reshape((ds_all.shape[0], -1))
    logging.info(f"Preprocessing: Concatenated centered data has shape {my_data.shape}")

    seq_data = np.zeros(
        (my_data.shape[0] - config.seq_len, config.seq_len, my_data.shape[1])
    )
    for i in range((ds_all.shape[0] - config.seq_len)):
        seq_data[i] = my_data[i : i + config.seq_len]
    logging.info(f"Preprocessing: Load seq_data of shape {seq_data.shape}")

    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Augmented seq_data has shape: {seq_data.shape}")

    five_perc = int(round(seq_data.shape[0] * 0.05))
    ninety_perc = seq_data.shape[0] - (2 * five_perc)
    train_ds = seq_data[:ninety_perc, :, :]
    valid_ds = seq_data[ninety_perc : (ninety_perc + five_perc), :, :]
    test_ds = seq_data[(ninety_perc + five_perc) :, :, :]
    logging.info(f">> Train ds has shape {train_ds.shape}")
    logging.info(f">> Valid ds has shape {valid_ds.shape}")
    logging.info(f">> Test ds has shape {test_ds.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")
    data_train_torch = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size
    )
    data_valid_torch = torch.utils.data.DataLoader(
        valid_ds, batch_size=config.batch_size
    )
    data_test_torch = torch.utils.data.DataLoader(test_ds, batch_size=1)
    return data_train_torch, data_valid_torch, data_test_torch


def sequify_data(pose_data, seq_len, augmentation_factor):
    seq_data = np.zeros(
        (pose_data.shape[0] - seq_len, seq_len, pose_data.shape[1])
    )

    for i in range((pose_data.shape[0] - seq_len)):
        seq_data[i] = pose_data[i : i + seq_len]
    logging.info(f"Preprocessing: Load seq_data_lab of shape {seq_data.shape}")

    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Augmented seq_data_lab has shape: {seq_data.shape}")

    return seq_data

def get_dgm_data(config, augmentation_factor=1):
    """Transform mariel data into train/val/test torch loaders.

    Note: Pettee 2019 keeps augmentation_factor=1.
    """
    ds_all, ds_all_centered, _, _, _ = load_mariel_raw()
    pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))


    
    labels, last_label_index, labelled_seq_len = load_labels()
     
    #divide into labelled and unlabelled
    
    pose_data_lab = pose_data[0:last_label_index]
    pose_data_unlab = pose_data[last_label_index:pose_data.shape[0]]

    #sequify both sets of data
    seq_data_lab = sequify_data(pose_data_lab, labelled_seq_len, augmentation_factor)
    seq_data_unlab = sequify_data(pose_data_unlab, labelled_seq_len, augmentation_factor)

    #divide into training, validating, and testing sets (associated labels?)
    
    five_perc = int(round(seq_data.shape[0] * 0.05))
    ninety_perc = seq_data.shape[0] - (2 * five_perc)
    labelled_train_ds = seq_data[:ninety_perc, :, :]
    
    valid_ds = seq_data[ninety_perc : (ninety_perc + five_perc), :, :]
    test_ds = seq_data[(ninety_perc + five_perc) :, :, :]


    logging.info(f">> Labelled Train ds has shape {labelled_data_train_ds.shape}")
    logging.info(f">> Unlabelled Train ds has shape {unlabelled_data_train_ds.shape}")
    logging.info(f">> Labelled Validation ds has shape {labelled_data_valid_ds.shape}")
    logging.info(f">> Labelled Test ds has shape {labelled_data_test_ds.shape}")
    logging.info(f">> Unlabelled Test ds has shape {unlabelled_data_test_ds.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")
    labelled_data_train = torch.utils.data.DataLoader(
        labelled_train_ds, batch_size=config.batch_size
    )


    labelled_data_train = torch.utils.data.DataLoader(labelled_data_train_ds, batch_size=config.batch_size)
    unlabelled_data_train = torch.utils.data.DataLoader(unlabelled_data_train_ds, batch_size=config.batch_size)
    labelled_data_valid = torch.utils.data.DataLoader(labelled_data_valid_ds, batch_size=config.batch_size)
    labelled_data_test = torch.utils.data.DataLoader(labelled_data_test_ds, batch_size=1)
    unlabelled_data_test = torch.utils.data.DataLoader(unlabelled_data_test_ds, batch_size=1)

    return labelled_data_train,unlabelled_data_train,labelled_data_valid,labelled_data_test,unlabelled_data_test



def get_aist_data(config, augmentation_factor=1):
    """Transform AIST++ data into train/val/test torch loaders.
    """

    ds_all, ds_all_centered, _, _, _ = load_aist_raw()
    my_data = ds_all_centered.reshape((ds_all.shape[0], -1))
    logging.info(f"Preprocessing: Concatenated centered data has shape {my_data.shape}")

    seq_data = np.zeros(
        (my_data.shape[0] - config.seq_len, config.seq_len, my_data.shape[1])
    )
    for i in range((ds_all.shape[0] - config.seq_len)):
        seq_data[i] = my_data[i : i + config.seq_len]
    logging.info(f"Preprocessing: Load seq_data of shape {seq_data.shape}")

    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Augmented seq_data has shape: {seq_data.shape}")

    five_perc = int(round(seq_data.shape[0] * 0.05))
    ninety_perc = seq_data.shape[0] - (2 * five_perc)
    train_ds = seq_data[:ninety_perc, :, :]
    valid_ds = seq_data[ninety_perc : (ninety_perc + five_perc), :, :]
    test_ds = seq_data[(ninety_perc + five_perc) :, :, :]
    logging.info(f">> Train ds has shape {train_ds.shape}")
    logging.info(f">> Valid ds has shape {valid_ds.shape}")
    logging.info(f">> Test ds has shape {test_ds.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")
    data_train_torch = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size
    )
    data_valid_torch = torch.utils.data.DataLoader(
        valid_ds, batch_size=config.batch_size
    )
    data_test_torch = torch.utils.data.DataLoader(test_ds, batch_size=1)
    return data_train_torch, data_valid_torch, data_test_torch

