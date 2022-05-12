# import numpy as np

# filepath="/home/papillon/move/move/data/labels.csv"
# file = open(filepath)
# labels_with_index = np.loadtxt(file, delimiter=",")

# labels = np.delete(labels_with_index, 0, axis=1)
# last_label_index = int(labels_with_index[-1][0])
# labelled_seq_len =  last_label_index - int(labels_with_index[-2][0])

import logging

import datasets
import default_config as config
import torch

ds_all, ds_all_centered, _, _, _ = datasets.load_mariel_raw()
pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))

labels, last_label_index, labelled_seq_len = datasets.load_labels()
print(labelled_seq_len)
print(labels.shape)

# divide into labelled and unlabelled

pose_data_lab = pose_data[0:last_label_index]
print("shape of pose data lab")
print(pose_data_lab.shape)
pose_data_unlab = pose_data[last_label_index : pose_data.shape[0]]

# sequify both sets of data
seq_data_lab = datasets.sequify_lab_data(
    pose_data_lab, labelled_seq_len, augmentation_factor=1
)
seq_data_unlab = datasets.sequify_all_data(
    pose_data, labelled_seq_len, augmentation_factor=1
)  # WE"RE SENDING IN SEQ THAT HAVE LABELS, just without the labels

# divide labelled data into 90% training, 5% validating, and 5% testing sets
five_perc_lab = int(round(seq_data_lab.shape[0] * 0.05))
ninety_perc_lab = seq_data_lab.shape[0] - (2 * five_perc_lab)
labelled_data_train_ds = seq_data_lab[:ninety_perc_lab, :, :]
labelled_data_valid_ds = seq_data_lab[
    ninety_perc_lab : (ninety_perc_lab + five_perc_lab), :, :
]
labelled_data_test_ds = seq_data_lab[(ninety_perc_lab + five_perc_lab) :, :, :]

# divide labels into 90% training, 5% validating, and 5% testing sets
labels_train_ds = labels[:ninety_perc_lab, :]
labels_valid_ds = labels[ninety_perc_lab : (ninety_perc_lab + five_perc_lab), :]
labels_test_ds = labels[(ninety_perc_lab + five_perc_lab) :, :]

# divide unlabelled data into 95% training and 5% testing sets (associated labels?)
five_perc_unlab = int(round(seq_data_unlab.shape[0] * 0.05))
ninety_perc_unlab = seq_data_unlab.shape[0] - (2 * five_perc_unlab)
unlabelled_data_train_ds = seq_data_unlab[: (ninety_perc_unlab + five_perc_unlab), :, :]
unlabelled_data_test_ds = seq_data_unlab[(ninety_perc_unlab + five_perc_unlab) :, :, :]

print(f">> Labelled Train ds has shape {labelled_data_train_ds.shape}")
print(f">> Unlabelled Train ds has shape {unlabelled_data_train_ds.shape}")
print(f">> Labelled Validation ds has shape {labelled_data_valid_ds.shape}")
print(f">> Labelled Test ds has shape {labelled_data_test_ds.shape}")
print(f">> Unlabelled Test ds has shape {unlabelled_data_test_ds.shape}")
print(f">> Labels train ds has shape {labels_train_ds.shape}")
print(f">> Labels valid ds has shape {labels_valid_ds.shape}")
print(f">> Labels test ds has shape {labels_test_ds.shape}")

labelled_data_train = torch.utils.data.DataLoader(
    labelled_data_train_ds, batch_size=config.batch_size
)
labels_train = torch.utils.data.DataLoader(
    labels_train_ds, batch_size=config.batch_size
)
unlabelled_data_train = torch.utils.data.DataLoader(
    unlabelled_data_train_ds, batch_size=config.batch_size
)
labelled_data_valid = torch.utils.data.DataLoader(
    labelled_data_valid_ds, batch_size=config.batch_size
)
labels_valid = torch.utils.data.DataLoader(
    labels_valid_ds, batch_size=config.batch_size
)
labelled_data_test = torch.utils.data.DataLoader(labelled_data_test_ds, batch_size=1)
labels_test = torch.utils.data.DataLoader(labels_test_ds, batch_size=config.batch_size)
unlabelled_data_test = torch.utils.data.DataLoader(
    unlabelled_data_test_ds, batch_size=1
)
