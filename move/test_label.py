# import numpy as np

# filepath="/home/papillon/move/move/data/labels.csv"
# file = open(filepath)
# labels_with_index = np.loadtxt(file, delimiter=",")

# labels = np.delete(labels_with_index, 0, axis=1)
# last_label_index = int(labels_with_index[-1][0])
# labelled_seq_len =  last_label_index - int(labels_with_index[-2][0])

import datasets
import default_config
import logging
import torch
import default_config as config
import numpy as np
import csv

labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, \
    labels_valid, labelled_data_test, labels_test, unlabelled_data_test = \
    datasets.get_dgm_data(default_config)

# ds_all, ds_all_centered, _, _, _ = datasets.load_mariel_raw()
# pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))

# labels, labels_ind  = datasets.load_labels()

# labels = labels.reshape((labels.shape[0], 1, labels.shape[-1]))

#     # divide into labelled and unlabelled

# # pose_data_lab = pose_data[0:last_label_index]
# # pose_data_unlab = pose_data[last_label_index:pose_data.shape[0]]

# # sequify both sets of data
# seq_data_lab = datasets.sequify_lab_data(labels_ind, pose_data, config.seq_len, augmentation_factor=1)
# seq_data_unlab = datasets.sequify_all_data(pose_data, config.seq_len, augmentation_factor=1) #WE"RE SENDING IN SEQ THAT HAVE LABELS, just without the labels

# # divide labelled data into 90% training, 5% validating, and 5% testing sets
# one_perc_lab = int(round(len(labels_ind) * 0.01))
# five_perc_lab = int(one_perc_lab * 5)
# # ninety_perc_lab = seq_data_lab.shape[0] - (2 * five_perc_lab)

# labelled_data_valid_ds = seq_data_lab[:(five_perc_lab), :, :]
# labelled_data_train_ds = seq_data_lab[(five_perc_lab) : ((five_perc_lab * 19) + (one_perc_lab * 3)), :, :]
# labelled_data_test_ds = seq_data_lab[((five_perc_lab * 19) + (one_perc_lab * 3)) :, :, :]

# # labelled_data_train_ds = seq_data_lab[:ninety_perc_lab, :, :]
# # labelled_data_valid_ds = seq_data_lab[ninety_perc_lab : (ninety_perc_lab + five_perc_lab), :, :]
# # labelled_data_test_ds = seq_data_lab[(ninety_perc_lab + five_perc_lab) :, :, :]

# #divide labels into 90% training, 5% validating, and 5% testing sets

# labels_valid_ds = labels[:(five_perc_lab * 2), :, :]
# labels_train_ds = labels[(five_perc_lab * 2) : ((five_perc_lab * 18) + (one_perc_lab * 8)), :, :]
# labels_test_ds = labels[((five_perc_lab * 18) + (one_perc_lab * 8)) :, :, :]


# # labels_train_ds = labels[:ninety_perc_lab, :]
# # labels_valid_ds = labels[ninety_perc_lab : (ninety_perc_lab + five_perc_lab), :]
# # labels_test_ds = labels[(ninety_perc_lab + five_perc_lab) :, :]

# # divide unlabelled data into 95% training and 5% testing sets (associated labels?)
# five_perc_unlab = int(round(seq_data_unlab.shape[0] * 0.05))
# ninety_perc_unlab = seq_data_unlab.shape[0] - (2 * five_perc_unlab)
# unlabelled_data_train_ds = seq_data_unlab[:(ninety_perc_unlab + five_perc_unlab), :, :]
# unlabelled_data_test_ds = seq_data_unlab[(ninety_perc_unlab + five_perc_unlab) :, :, :]

alpha = 0.01 * len(unlabelled_data_train) / len(labelled_data_train)
print(alpha)