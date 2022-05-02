# import numpy as np

# filepath="/home/papillon/move/move/data/labels.csv"
# file = open(filepath)
# labels_with_index = np.loadtxt(file, delimiter=",")

# labels = np.delete(labels_with_index, 0, axis=1)
# last_label_index = int(labels_with_index[-1][0])
# labelled_seq_len =  last_label_index - int(labels_with_index[-2][0])

import datasets


ds_all, ds_all_centered, _, _, _ = datasets.load_mariel_raw()
pose_data = ds_all_centered.reshape((ds_all.shape[0], -1))

labels, last_label_index, labelled_seq_len = datasets.load_labels()


#divide into labelled and unlabelled

pose_data_lab = pose_data[0:last_label_index]
pose_data_unlab = pose_data[last_label_index:pose_data.shape[0]]

#sequify both sets of data
seq_data_lab = datasets.sequify_data(pose_data_lab, labelled_seq_len, augmentation_factor=1)
seq_data_unlab = datasets.sequify_data(pose_data_unlab, labelled_seq_len, augmentation_factor=1)

print(seq_data_lab.shape)
print(seq_data_unlab.shape)