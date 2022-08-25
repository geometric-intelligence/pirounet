"""Utils to process raw data and make them into tensors."""

import logging
import os
import pickle
from glob import glob

import default_config
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset


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
    seq_data : array
        Shape = [n_seqs, seq_len, input_features]
        Original dataset of sequences.
    augmentation_factor : int
        Multiplication factor, i.e. how many new sequences
        are generated from one given sequence.

    Returns
    -------
    rotated_seq_data : array-like
        Shape = [augmentation_factor*n_seqs, seq_len,
                input_features]
        Augmented dataset of sequences.
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


def load_raw(pattern="data/mariel_*.npy"):
    """Load six datasets and perform minimal preprocessing.

    Processing amunts to center each dancer, such that
    the barycenter becomes 0.

    From Pettee 2019:
    Each frame of the dataset is transformed such that the
    overall average (x,y) position per frame is centered at
    the same point and scaled such that all of the coordinates
    fit within the unit cube.

    Parameters
    ----------
    pattern : string
        Path to the raw datafiles.

    Returns
    -------
    ds_all : array
        Shape = [n_seqs, seq_len, input_features]
        Concatenated dance sequences from all datafiles.
    ds_all_centered : array
        Shape = [n_seqs, seq_len, input_features]
        Concatenated dance sequences from all datafiles.
        with all datasets centered, i.e. with the x and
        y offsets subtracted from each individual frame.
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

        datasets[ds_name] = ds
        ds_all.append(ds)

    ds_all = np.concatenate(ds_all)
    low, hi = np.quantile(ds_all, [0.01, 0.99], axis=(0, 1))
    xy_min = min(low[:2])
    xy_max = max(hi[:2])
    xy_range = xy_max - xy_min
    ds_all[:, :, :2] -= xy_min
    ds_all *= 2 / xy_range
    ds_all[:, :, :2] -= 1.0

    ds_all_centered = ds_all.copy()
    ds_all_centered[:, :, :2] -= ds_all_centered[:, :, :2].mean(axis=1, keepdims=True)

    return ds_all, ds_all_centered


def load_labels(
    effort,
    filepath="/home/papillon/move/move/data/labels_from_app.csv",
    no_NA=True,
    augment=True,
):
    """Load labels created by human.

    The web labeling app saves labels in a csv file,
    in the order of appearance of the sequences in
    the entire dataset.

    Parameters
    ----------
    effort : string
        Corresponds to the specific category of label
        the model will be trained on. We use 'time'
        for Laban Time Effort intensity and 'space'
        for Laban Space Effort intensity.
    filepath : string
        Path to the csv file containing manual labels.
        Must be moved there from the web-app directory.
    no_NA : bool
        True: includes sequences that were labeled as
        "Non-Applicable", i.e. integer 4.
        False: excludes these sequences.
    augment : bool
        True: Uses label augmentation tool to increase
        labeled sequences automatically
        False: Only loads manual labels

    Returns
    -------
    labels : array
        Shape = [n_seqs_labeled, 1]
        Array of labels assosicated to labeled sequences,
        stored as an integer for each categorical label.
    labels_ind : array
        Shape = [n_seqs_labeled, 1]
        Array of sequence indices assosicated to labeled
        sequences (pose index of the labeled sequence's
        first pose).
    """

    file = open(filepath)
    labels_with_index = np.loadtxt(file, delimiter=",")

    if effort == "time":
        labels_with_index = np.delete(labels_with_index, 1, axis=1)

    if effort == "space":
        labels_with_index = np.delete(labels_with_index, 2, axis=1)

    if augment:
        labels_with_index = augment_labels(labels_with_index)

    if no_NA:
        labels_with_index_noNA = []
        for i in range(len(labels_with_index)):
            row = labels_with_index[i]
            if row[1] != 4.0:
                labels_with_index_noNA.append(row)

        labels_ind = np.delete(labels_with_index_noNA, 1, axis=1)
        labels = np.delete(labels_with_index_noNA, 0, axis=1)

    if not no_NA:
        labels_ind = np.delete(labels_with_index, 1, axis=1)
        labels = np.delete(labels_with_index, 0, axis=1)

    return labels, labels_ind


def augment_labels(labels_with_index, seq_len=default_config.seq_len):
    """Augment labels created by human.

    Tool for automatically increasing the amount of manually
    created labels, called when labels are loaded. Performs
    two augmentations:
    1. Label all sequences between sequences that share a same Effort.
    For example, if two back-to-back sequences are deemed to have Low
    Time Effort, all sequences that are a combination of the poses in
    these two sequences are also labeled with Low Time effort.
    2. Extend every label to all sequences starting within 6 frames
    (0.17 seconds) before or after its respective sequence.

    Parameters
    ----------
    labels_with_index : array
        Shape = [n_seqs_labeled, 2]
        Array of manually labeled sequence indices with associated
        manually created labels.
    seq_len : int
        Amount of poses in a sequence.

    Returns
    -------
    labels_with_index_aug : array
        Shape = [n_augmented_seqs_labeled, 2]
        Array of manually and automatically labeled sequence indices
        with associated manually and automatically created labels.
    """

    all_between_lab = np.zeros((1, 2))

    # Label sequences between two identically labelled block sequences
    for j in range(len(labels_with_index)):
        if j == 0:
            continue

        if j != 0:
            bef_index = labels_with_index[j - 1][0]
            effort = labels_with_index[j][1]
            bef_effort = labels_with_index[j - 1][1]

            if effort == bef_effort:
                between_lab = np.zeros((int(seq_len), 2))

                for i in range(int(seq_len)):
                    between_lab[i][0] = int(bef_index + i + 1)
                    between_lab[i][1] = int(effort)
                    between_lab = np.array(between_lab).reshape((-1, 2))

                all_between_lab = np.append(all_between_lab, between_lab, axis=0)
    all_between_lab = all_between_lab[
        1:,
    ]

    # Label sequences starting 6 poses before and after each block sequence
    extra_frames = 6
    fuzzy_labels = np.zeros((1, 2))

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
                extra_label_neg = np.expand_dims(
                    [index_labelled - (i_rev + 1), effort], axis=0
                )
                fuzzy_labels = np.append(fuzzy_labels, extra_label_neg, axis=0)

            fuzzy_labels = np.append(
                fuzzy_labels, labels_with_index[j].reshape((1, 2)), axis=0
            )

            for i in range(extra_frames + 1):
                extra_label_pos = np.expand_dims(
                    [index_labelled + (i + 1), effort], axis=0
                )
                fuzzy_labels = np.append(fuzzy_labels, extra_label_pos, axis=0)
    fuzzy_labels = fuzzy_labels[
        1:,
    ]

    nonunique = np.append(all_between_lab, fuzzy_labels, axis=0)

    labels_with_index_aug = nonunique[
        np.unique(nonunique[:, 0], axis=0, return_index=True)[1]
    ]
    return labels_with_index_aug


def sequify_all_data(pose_data, seq_len, augmentation_factor):
    """Cut all pose data into sequences.

    Stores every possible continuous seq_long-long sequence from
    the concatenated pose data. Makes n_poses - seq_len total
    sequences.

    Parameters
    ----------
    pose_data : array
        Shape = [n_poses, input_dim]
        Every single pose in the dataset, in order. Does not
        take into account which ones have been labeled.
        Input dimensions are amount_of_keypoints * 3 (for
        3D coordinates)
    seq_len : int
        Amount of poses in a sequence.
    augmentation_factor : int
        Determines rotational augmentation of data.
        Independent from label augmentation.

    Returns
    -------
    seq_data : array
        Shape = [n_seqs, seq_len, input_dim]
        Randomly shuffled array of sequences made
        from pose data. Includes labeled and
        unlabeled sequences
    """
    seq_data = np.zeros((pose_data.shape[0] - seq_len, seq_len, pose_data.shape[1]))

    for i in range((pose_data.shape[0] - seq_len)):
        seq_data[i] = pose_data[i : i + seq_len]
    logging.info(f"Preprocessing: Load seq_data of shape {seq_data.shape}")

    # used to augment here

    # np.random.shuffle(seq_data)

    return seq_data


def sequify_lab_data(labels_ind, pose_data, seq_len, augmentation_factor):
    """Cut labeled pose data into sequences.

    Uses indices of labeled sequences to store sequences that were
    labeled manually (and automatically, if augmentation used).

    Parameters
    ----------
    labels_ind : array
        Shape = [n_aug_seq_labeled, 1] or [n_seq_labeled, 1]
        Indices of first pose associated to each labeled sequence.
    pose_data : array
        Shape = [n_poses_labeled, input_dim]
        Every single labeled pose in the dataset, in order.
        Input dimensions are amount_of_keypoints * 3 (for
        3D coordinates)
    seq_len : int
        Amount of poses in a sequence.
    augmentation_factor : int
        Determines rotational augmentation of data.
        Independent from label augmentation.

    Returns
    -------
    seq_data : array
        Shape = [n_seqs_labeled, seq_len, input_dim]
        Array of labeled sequences made from pose data.
    """
    seq_data = np.zeros((len(labels_ind), seq_len, pose_data.shape[1]))
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

    return seq_data


def array_from_sparse(indices, data, target_shape):
    """Create an array of given shape, with values at specific indices.
    The rest of the array will be filled with -1.
    Parameters
    ----------
    indices : iterable(tuple(int))
        Index of each element which will be assigned a specific value.
    data : iterable(scalar)
        Value associated at each index.
    target_shape : tuple(int)
        Shape of the output array.
    Returns
    -------
    a : array, shape=target_shape
        Array of zeros with specified values assigned to specified indices.
    """
    out = -1 * np.ones(target_shape)
    for ind, val in zip(indices, data):
        out[int(ind)] = val
    return out


def get_model_data(config):
    """Transforms raw data for training/validating/testing a model.

    Pipeline for loading raw data as torch DataLoaders,
    separated intro training, validation, and test datasets.
    The proportion of training labeled data is set by param
    fraction_label in config. The first 5% is reserved for
    validation. The last 3% is reserved for testing.
    For the unalbeled case, we reserve 95% of sequences for
    training and 5% for testing.
    Note: Pettee 2019 keeps augmentation_factor=1.

    Parameters
    ----------
    config : dict
        Configuration of run as defined in wandb.config

    Returns
    -------
    labelled_data_train : array
        Shape = [n_seqs_labeled * fraction_label, seq_len, input_dim]
        Array of labeled sequences made from pose data
        reserved for training.
    labels_train : array
        Shape = [n_seqs_labeled * fraction_label, 1]
        Labels associated to training labeled sequences.
    unlabelled_data_train : array
        Shape = [n_seqs (* 0.95), seq_len, input_dim]
        Array of unlabeled sequences made from pose data
        reserved for training.
    labelled_data_valid : array
        Shape = [n_seqs_labeled * 0.05, seq_len, input_dim]
        Array of labeled sequences made from pose data
        reserved for validation.
    labels_valid : array
        Shape = [n_seqs_labeled * 0.05, 1]
        Labels associated to validation labeled sequences.
    labelled_data_test : array
        Shape = [n_seqs_labeled * 0.03, seq_len, input_dim]
        Array of labeled sequences made from pose data
        reserved for testing.
    labels_test : array
        Shape = [n_seqs_labeled * 0.03, 1]
        Labels associated to testing labeled sequences.
    unlabelled_data_test : array
        Shape = [n_seqs * 0.05, seq_len, input_dim]
        Array of unlabeled sequences made from pose data
        reserved for testing.
    """

    ds_all, ds_all_centered = load_raw()
    pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))

    labels_1_to_4_vals, labels_ind = load_labels(effort=config.effort)
    labels_vals = labels_1_to_4_vals - 1.0
    labels_vals = labels_vals.reshape((labels_vals.shape[0], 1, labels_vals.shape[-1]))

    # # sequify both sets of data
    # seq_data_lab = sequify_lab_data(
    #     labels_ind, pose_data, config.seq_len, augmentation_factor=1
    # )
    seq_data = sequify_all_data(pose_data, config.seq_len, augmentation_factor=1)

    # augment here now
    augmentation_factor = 1
    if augmentation_factor > 1:
        logging.info(
            "Preprocessing: data augmentation by rotations, "
            f"factor = {augmentation_factor}"
        )
        seq_data = augment_by_rotations(seq_data, augmentation_factor)
        logging.info(f">> Unlabelled augmented seq_data has shape: {seq_data.shape}")
        aug_labels_ind = []
        for l in labels_ind:
            for i in range(augmentation_factor):
                aug_labels_ind.append(augmentation_factor * l + i)
        labels_ind = aug_labels_ind

    # create labels array corresponding directly to all sequences
    labels = array_from_sparse(labels_ind, labels_vals, (seq_data.shape[0],))

    # # calculate chunk of unlab data to remove for random sampling purposes
    # n_remove = len(seq_data_unlab) - len(labels_vals) / config.valid_frac_labels
    # seq_data_start = seq_data_unlab[: - n_remove, :, :]
    # labels_start = labels[: - n_remove]

    # divide into validation / train / test
    seq_data_train, seq_data_val_test, labels_train, labels_val_test = train_test_split(
        seq_data, labels, test_size=(1 - config.frac_train), random_state=42
    )  # test_size is percentage given to test+valid

    seq_data_val, seq_data_test, labels_val, labels_test = train_test_split(
        seq_data_val_test, labels_val_test, test_size=0.25, random_state=42
    )  # size of test set compared to validation+test set

    # modify fraction of labelled to unlab train
    n_labels_in_train = np.sum([lab != -1 for lab in labels_train])
    # current_frac = n_labels_in_train / len(labels_train)
    n_labels_target = len(labels_train) * config.train_lab_frac
    n_labels_to_remove = n_labels_in_train - n_labels_target

    if n_labels_to_remove < 0:
        raise ValueError("va te faire foutre.")

    for i_lab, lab in enumerate(labels_train):
        if lab != -1:
            labels_train[i_lab] = -1
            n_labels_to_remove -= 1
            if n_labels_to_remove == 0:
                break

    seq_data_train_labelled = seq_data_train[labels_train != -1]
    labels_train = labels_train[labels_train != -1]
    seq_data_val_labelled = seq_data_val[labels_val != -1]
    labels_val = labels_val[labels_val != -1]
    seq_data_test_labelled = seq_data_test[labels_test != -1]
    labels_test = labels_test[labels_test != -1]

    seq_data_train_unlab = seq_data_train
    seq_data_val_unlab = seq_data_val
    seq_data_test_unlab = seq_data_test

    # # divide labelled data into training, validating, and testing sets
    # one_perc_lab = int(round(len(labels_ind) * 0.01))
    # five_perc_lab = int(one_perc_lab * 5)
    # new_stopping_point = (
    #     int(round(len(labels_ind) * config.fraction_label)) + five_perc_lab
    # )

    # labelled_data_valid_ds = seq_data_lab[:(five_perc_lab), :, :]
    # labelled_data_train_ds = seq_data_lab[(five_perc_lab):new_stopping_point, :, :]
    # labelled_data_test_ds = seq_data_lab[
    #     ((five_perc_lab * 19) + (one_perc_lab * 2)) :, :, :
    # ]

    # # divide labels into training, validating, and testing sets
    # labels_valid_ds = labels[:(five_perc_lab), :, :]
    # labels_train_ds = labels[(five_perc_lab):new_stopping_point, :, :]
    # labels_test_ds = labels[((five_perc_lab * 19) + (one_perc_lab * 2)) :, :, :]

    # # divide unlabelled data into training and testing sets
    # five_perc_unlab = int(round(seq_data_unlab.shape[0] * 0.05))
    # ninety_perc_unlab = seq_data_unlab.shape[0] - (2 * five_perc_unlab)
    # unlabelled_data_train_ds = seq_data_unlab[
    #     : (ninety_perc_unlab + five_perc_unlab), :, :
    # ]
    # unlabelled_data_test_ds = seq_data_unlab[
    #     (ninety_perc_unlab + five_perc_unlab) :, :, :
    # ]

    logging.info(f">> Labelled Train ds has shape {seq_data_train_labelled.shape}")
    logging.info(f">> Unlabelled Train ds has shape {seq_data_train_unlab.shape}")
    logging.info(f">> Labelled Validation ds has shape {seq_data_val_labelled.shape}")
    logging.info(f">> Unlabelled Validation ds has shape {seq_data_val_unlab.shape}")
    logging.info(f">> Labelled Test ds has shape {seq_data_test_labelled.shape}")
    logging.info(f">> Unlabelled Test ds has shape {seq_data_test_unlab.shape}")
    logging.info(f">> Labels train ds has shape {labels_train.shape}")
    logging.info(f">> Labels valid ds has shape {labels_val.shape}")
    logging.info(f">> Labels test ds has shape {labels_test.shape}")

    logging.info("Preprocessing: Convert into torch dataloader")

    labelled_data_train = torch.utils.data.DataLoader(
        seq_data_train_labelled, batch_size=config.batch_size, drop_last=True
    )
    labels_train = torch.utils.data.DataLoader(
        labels_train, batch_size=config.batch_size, drop_last=True
    )
    unlabelled_data_train = torch.utils.data.DataLoader(
        seq_data_train_unlab, batch_size=config.batch_size
    )
    labelled_data_valid = torch.utils.data.DataLoader(
        seq_data_val_labelled, batch_size=config.batch_size, drop_last=True
    )
    labels_valid = torch.utils.data.DataLoader(
        labels_val, batch_size=config.batch_size, drop_last=True
    )
    labelled_data_test = torch.utils.data.DataLoader(
        seq_data_test_labelled, batch_size=1, drop_last=True
    )
    labels_test = torch.utils.data.DataLoader(
        labels_test, batch_size=config.batch_size, drop_last=True
    )
    unlabelled_data_test = torch.utils.data.DataLoader(
        seq_data_test_unlab,
        batch_size=1,
    )

    return (
        labelled_data_train,
        labels_train,
        unlabelled_data_train,
        labelled_data_valid,
        labels_valid,
        labelled_data_test,
        labels_test,
        unlabelled_data_test,
    )
