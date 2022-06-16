"""Utils to process raw data and make them into tensors."""

import logging
import os
import pickle
from glob import glob

import numpy as np
import torch
from torch.utils.data import ConcatDataset

import default_config

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
        # ds = filter_points(ds)

        datasets[ds_name] = ds
        ds_all.append(ds)

    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1])

    ds_all = np.concatenate(ds_all)

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

    low, hi = np.quantile(ds_all, [0.01, 0.99], axis=(0, 1))
    return ds_all, ds_all_centered, datasets, datasets_centered, ds_counts


def load_labels(
    effort,
    filepath="/home/papillon/move/move/data/labels_mathilde.csv",
    no_NA=True,
    augment=True
):

    file = open(filepath)
    labels_with_index = np.loadtxt(file, delimiter=",")

    if effort == 'time':
        labels_with_index = np.delete(labels_with_index, 1, axis=1)
        
    if effort == 'space':
        labels_with_index = np.delete(labels_with_index, 2, axis=1)

    if augment:
        labels_with_index = augment_labels(labels_with_index)

    if no_NA:
        labels_with_index_noNA = []
        for i in range(len(labels_with_index)):
            row = labels_with_index[i]
            if row[1] != 4.:
                labels_with_index_noNA.append(row)

        labels_ind = np.delete(labels_with_index_noNA, 1, axis=1)
        labels = np.delete(labels_with_index_noNA, 0, axis=1)
    
    if not no_NA:
        labels_ind = np.delete(labels_with_index, 1, axis=1)
        labels = np.delete(labels_with_index, 0, axis=1)


    return labels, labels_ind


def augment_labels(labels_with_index, seq_len=default_config.seq_len):
    all_between_lab = np.zeros((1,2))

    # Label sequences between two identically labelled block sequences
    for j in range(len(labels_with_index)):
        if j == 0:
            continue

        if j != 0:
            bef_index = labels_with_index[j-1][0]
            effort = labels_with_index[j][1]
            bef_effort = labels_with_index[j-1][1]

            if effort == bef_effort:
                between_lab = np.zeros((int(seq_len), 2))

                for i in range(int(seq_len)):
                    between_lab[i][0] = int(bef_index + i + 1)
                    between_lab[i][1] = int(effort)
                    between_lab = np.array(between_lab).reshape((-1,2))

                all_between_lab = np.append(all_between_lab, between_lab, axis=0)
    all_between_lab = all_between_lab[1:,]
    
    # Label sequences starting a few poses before and after each block sequence
    extra_frames = 5
    fuzzy_labels = np.zeros((1,2))

    for j in range(len(labels_with_index)):
        index_labelled = labels_with_index[j][0]
        effort = labels_with_index[j][1]

        if index_labelled == 0:
            for i in range(extra_frames + 2):
                extra_label = np.expand_dims([i, effort], axis=0)
                fuzzy_labels = np.append(fuzzy_labels, extra_label, axis=0)

        if index_labelled != 0:
            for i in range(extra_frames + 1):
                i_rev = extra_frames - i
                extra_label_neg = np.expand_dims([index_labelled - (i_rev + 1), effort], axis=0)
                fuzzy_labels = np.append(fuzzy_labels, extra_label_neg, axis=0)

            fuzzy_labels = np.append(fuzzy_labels, labels_with_index[j].reshape((1,2)), axis=0)

            for i in range(extra_frames + 1):
                extra_label_pos = np.expand_dims([index_labelled + (i + 1), effort], axis=0)
                fuzzy_labels = np.append(fuzzy_labels, extra_label_pos, axis=0)
    fuzzy_labels = fuzzy_labels[1:,]

    nonunique = np.append(all_between_lab, fuzzy_labels, axis=0)

    labels_with_index_aug = nonunique[np.unique(nonunique[:,0], axis=0, return_index=True)[1]]

    # with open('smart_labels_mathilde.csv', 'w', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(labels_with_index_aug)
    # file.close()

    return labels_with_index_aug

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
        data = pickle.load(open(f, "rb"))

        kp3d = data["keypoints3d"]
        kp3d_optim = data["keypoints3d_optim"]
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

    # five_perc = int(round(seq_data.shape[0] * 0.05))
    # ninety_perc = seq_data.shape[0] - (2 * five_perc)
    # train_ds = seq_data[:ninety_perc, :, :]
    # valid_ds = seq_data[ninety_perc : (ninety_perc + five_perc), :, :]
    # test_ds = seq_data[(ninety_perc + five_perc) :, :, :]
    five_perc = int(round(seq_data.shape[0] * 0.05))
    ninety_perc = seq_data.shape[0] - (2 * five_perc)
    valid_ds = seq_data[:five_perc, :, :]
    test_ds = seq_data[five_perc:(five_perc + five_perc), :, :]
    train_ds = seq_data[(five_perc + five_perc):, :, :]

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


def sequify_all_data(pose_data, seq_len, augmentation_factor):
    seq_data = np.zeros(
        (pose_data.shape[0] - seq_len, seq_len, pose_data.shape[1])
    )

    for i in range((pose_data.shape[0] - seq_len)):
        seq_data[i] = pose_data[i : i + seq_len]
    logging.info(f"Preprocessing: Load seq_data of shape {seq_data.shape}")

    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Augmented seq_data has shape: {seq_data.shape}")

    np.random.shuffle(seq_data)

    return seq_data


def sequify_lab_data(labels_ind, pose_data, seq_len, augmentation_factor):
    seq_data = np.zeros(
        (len(labels_ind), seq_len, pose_data.shape[1])
    )
    for i in range(len(labels_ind)):
        start_ind = int(labels_ind[i])
        seq_data[i] = pose_data[start_ind : start_ind + int(seq_len)]

    logging.info(f"Preprocessing: Load labelled data of shape {seq_data.shape}")

    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Augmented labelled data has shape: {seq_data.shape}")
    
    #np.random.shuffle(seq_data)

    return seq_data

def get_dgm_data(config, augmentation_factor=1):
    """Transform mariel data into train/val/test torch loaders.

    Note: Pettee 2019 keeps augmentation_factor=1.
    """
    ds_all, ds_all_centered, _, _, _ = load_mariel_raw()
    pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))

    labels_1_to_4, labels_ind = load_labels(effort = config.effort)
    labels = labels_1_to_4 - 1.
    labels = labels.reshape((labels.shape[0], 1, labels.shape[-1]))

    # sequify both sets of data
    seq_data_lab = sequify_lab_data(labels_ind, pose_data, config.seq_len, augmentation_factor=1)
    seq_data_unlab = sequify_all_data(pose_data, config.seq_len, augmentation_factor=1)

    # divide labelled data into 90% training, 5% validating, and 5% testing sets
    one_perc_lab = int(round(len(labels_ind) * 0.01))
    five_perc_lab = int(one_perc_lab * 5)

    labelled_data_valid_ds = seq_data_lab[:(five_perc_lab), :, :]
    labelled_data_train_ds = seq_data_lab[(five_perc_lab) : ((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    labelled_data_test_ds = seq_data_lab[((five_perc_lab * 19) + (one_perc_lab * 2)) :, :, :]

    #double_test_ds = np.append(labelled_data_test_ds, labelled_data_test_ds)

    # labelled_data_valid_ds = seq_data_lab[(12*five_perc_lab):(13*five_perc_lab), :, :]
    # train1 = seq_data_lab[:(12*five_perc_lab), :, :]
    # train2 = seq_data_lab[(13*five_perc_lab):((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    # labelled_data_train_ds = np.append(train1, train2, axis=0)
    # labelled_data_test_ds = seq_data_lab[((five_perc_lab * 19) + (one_perc_lab * 3)) :, :, :]

    #divide labels into 90% training, 5% validating, and 5% testing sets
    labels_valid_ds = labels[:(five_perc_lab), :, :]
    labels_train_ds = labels[(five_perc_lab) : ((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    labels_test_ds = labels[((five_perc_lab * 19) + (one_perc_lab * 2)) :, :, :]

    #double_test_l_ds = np.append(labels_test_ds, labels_test_ds)
    # labels_valid_ds = labels[(12*five_perc_lab):(13*five_perc_lab), :, :]
    # train1l = labels[:(12*five_perc_lab), :, :]
    # train2l = labels[(13*five_perc_lab):((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    # labels_train_ds = np.append(train1l, train2l)
    # labels_test_ds = labels[((five_perc_lab * 19) + (one_perc_lab * 3)) :, :, :]

    # divide unlabelled data into 95% training and 5% testing sets (associated labels?)
    five_perc_unlab = int(round(seq_data_unlab.shape[0] * 0.05))
    ninety_perc_unlab = seq_data_unlab.shape[0] - (2 * five_perc_unlab)
    unlabelled_data_train_ds = seq_data_unlab[:(ninety_perc_unlab + five_perc_unlab), :, :]
    unlabelled_data_test_ds = seq_data_unlab[(ninety_perc_unlab + five_perc_unlab) :, :, :]

    print(f">> Labelled Train ds has shape {labelled_data_train_ds.shape}")
    print(f">> Unlabelled Train ds has shape {unlabelled_data_train_ds.shape}")
    print(f">> Labelled Validation ds has shape {labelled_data_valid_ds.shape}")
    print(f">> Labelled Test ds has shape {labelled_data_test_ds.shape}")
    print(f">> Unlabelled Test ds has shape {unlabelled_data_test_ds.shape}")
    print(f">> Labels train ds has shape {labels_train_ds.shape}")
    print(f">> Labels valid ds has shape {labels_valid_ds.shape}")
    print(f">> Labels test ds has shape {labels_test_ds.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")

    labelled_data_train = torch.utils.data.DataLoader(
        labelled_data_train_ds, batch_size=config.batch_size, drop_last=True
        )
    labels_train = torch.utils.data.DataLoader(
        labels_train_ds, batch_size=config.batch_size, drop_last=True
        )
    unlabelled_data_train = torch.utils.data.DataLoader(
        unlabelled_data_train_ds, batch_size=config.batch_size
        )
    labelled_data_valid = torch.utils.data.DataLoader(
        labelled_data_valid_ds, batch_size=config.batch_size, drop_last=True
        )
    labels_valid = torch.utils.data.DataLoader(
        labels_valid_ds, batch_size=config.batch_size, drop_last=True
        )
    labelled_data_test = torch.utils.data.DataLoader(
        labelled_data_test_ds, batch_size=1, drop_last=True
        )
    labels_test = torch.utils.data.DataLoader(
        labels_test_ds, batch_size=config.batch_size, drop_last=True
        )
    unlabelled_data_test = torch.utils.data.DataLoader(
        unlabelled_data_test_ds, batch_size=1,
        )
    # double_test = torch.utils.data.DataLoader(
    #     double_test_ds, batch_size=1,
    #     )

    return labelled_data_train, labels_train, unlabelled_data_train, \
            labelled_data_valid, labels_valid, labelled_data_test, labels_test,\
            unlabelled_data_test


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

def get_classifier_data(config, augmentation_factor=1):
    """Transform mariel data into train/val/test torch loaders.

    Note: Pettee 2019 keeps augmentation_factor=1.
    """
    ds_all, ds_all_centered, _, _, _ = load_mariel_raw()
    pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))

    labels_1_to_4, labels_ind = load_labels(effort = config.effort)
    labels = labels_1_to_4 - 1.
    labels = labels.reshape((labels.shape[0], 1, labels.shape[-1]))

    # sequify both sets of data
    seq_data_lab = sequify_lab_data(labels_ind, pose_data, config.seq_len, augmentation_factor=1)
    seq_data_unlab = sequify_all_data(pose_data, config.seq_len, augmentation_factor=1)

    # divide labelled data into 90% training, 5% validating, and 5% testing sets
    one_perc_lab = int(round(len(labels_ind) * 0.01))
    five_perc_lab = int(one_perc_lab * 5)

    labelled_data_valid_ds = seq_data_lab[:(five_perc_lab), :, :]
    labelled_data_train_ds = seq_data_lab[(five_perc_lab) : ((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    labelled_data_test_ds = seq_data_lab[((five_perc_lab * 19) + (one_perc_lab * 2)) :, :, :]

    #double_test_ds = np.append(labelled_data_test_ds, labelled_data_test_ds)

    # labelled_data_valid_ds = seq_data_lab[(12*five_perc_lab):(13*five_perc_lab), :, :]
    # train1 = seq_data_lab[:(12*five_perc_lab), :, :]
    # train2 = seq_data_lab[(13*five_perc_lab):((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    # labelled_data_train_ds = np.append(train1, train2, axis=0)
    # labelled_data_test_ds = seq_data_lab[((five_perc_lab * 19) + (one_perc_lab * 3)) :, :, :]

    #divide labels into 90% training, 5% validating, and 5% testing sets
    labels_valid_ds = labels[:(five_perc_lab), :, :]
    labels_train_ds = labels[(five_perc_lab) : ((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    labels_test_ds = labels[((five_perc_lab * 19) + (one_perc_lab * 2)) :, :, :]

    #double_test_l_ds = np.append(labels_test_ds, labels_test_ds)
    # labels_valid_ds = labels[(12*five_perc_lab):(13*five_perc_lab), :, :]
    # train1l = labels[:(12*five_perc_lab), :, :]
    # train2l = labels[(13*five_perc_lab):((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
    # labels_train_ds = np.append(train1l, train2l)
    # labels_test_ds = labels[((five_perc_lab * 19) + (one_perc_lab * 3)) :, :, :]

    logging.info("Get generated data")
    labelled_data_gen = np.load('data/shuffled_seq_for_classifier.npy')
    labelled_data_gen = labelled_data_gen.reshape(-1, config.seq_len, config.input_dim)
    labelled_data_train_ds = np.append(labelled_data_train,labelled_data_gen)
    labels_train_gen = np.load('data/shuffled_labels_for_classifier.npy')
    labels_train_gen = labelled_data_gen.reshape(-1, 1, 1)
    labels_train_ds = np.append(labels_train_ds, labels_train_gen)

    print(f">> Labelled Train ds has shape {labelled_data_train_ds.shape}")
    print(f">> Labelled Validation ds has shape {labelled_data_valid_ds.shape}")
    print(f">> Labelled Test ds has shape {labelled_data_test_ds.shape}")
    print(f">> Labels train ds has shape {labels_train_ds.shape}")
    print(f">> Labels valid ds has shape {labels_valid_ds.shape}")
    print(f">> Labels test ds has shape {labels_test_ds.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")

    labelled_data_train = torch.utils.data.DataLoader(
        labelled_data_train_ds, batch_size=config.batch_size, drop_last=True
        )
    labels_train = torch.utils.data.DataLoader(
        labels_train_ds, batch_size=config.batch_size, drop_last=True
        )
    labelled_data_valid = torch.utils.data.DataLoader(
        labelled_data_valid_ds, batch_size=config.batch_size, drop_last=True
        )
    labels_valid = torch.utils.data.DataLoader(
        labels_valid_ds, batch_size=config.batch_size, drop_last=True
        )
    labelled_data_test = torch.utils.data.DataLoader(
        labelled_data_test_ds, batch_size=1, drop_last=True
        )
    labels_test = torch.utils.data.DataLoader(
        labels_test_ds, batch_size=config.batch_size, drop_last=True
        )

    return labelled_data_train, labels_train, \
            labelled_data_valid, labels_valid, labelled_data_test, labels_test,\
            

