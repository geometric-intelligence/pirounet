"""File that generate a confusion matrix from given checkpoint. Specify train/valid/test"""

import os
import sys
sys.path.append(os. path. abspath('..'))

import default_config as config
import datasets
from models import dgm_lstm_vae

import torch
import torch.nn as nn
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

labels_before, labels_before_ind = datasets.load_labels(
    effort = 'time',
    filepath="/home/papillon/move/move/data/manual_labels.csv",
    no_NA=False,
    augment = True)

labels_now, labels_now_ind = datasets.load_labels(
    effort = 'time',
    filepath="/home/papillon/move/move/data/labels_mathilde.csv",
    no_NA=False,
    augment = True)


# conf_mat = ConfusionMatrixDisplay.from_predictions(
#     labels_before, 
#     labels_now,
#     #labels = ['Low', 'Medium', 'High', 'N/A'],
#     #cmap = 'Blues'
#     )

# purpose = 'self'
# plt.savefig(fname="../saved/confusion/conf_" + str(purpose) + ".png")

#Now we want to calculate my own test accuracy

# only compare elements that are the same. get array of true/false for each label, and in total

# boolarr = np.equal(labels_before, labels_now)
# num = boolarr.sum()
# frac = num / len(labels_before)
# print(frac)
# with 4 labels, i have test accuracy of 56%
#what about on first 3 labels?

labels_before, labels_before_ind = datasets.load_labels(
    effort = 'time',
    filepath="/home/papillon/move/move/data/manual_labels.csv",
    no_NA=True,
    augment = True)

labels_now, labels_now_ind = datasets.load_labels(
    effort = 'time',
    filepath="/home/papillon/move/move/data/labels_mathilde.csv",
    no_NA=True,
    augment = True)

labels_before = labels_before - 1.
labels_now = labels_now - 1.
#find the elements we have extra and remove them from each list

res = [i for i in labels_now_ind if i in labels_before_ind]
new_ind_now = []
new_ind_bef = []

for i in range(len(res)):
    ind = np.where(res[i]==labels_now_ind)
    new_ind_now.append(ind)
    ind_bef = np.where(res[i]==labels_before_ind)
    new_ind_bef.append(ind_bef)

labels_now_new = labels_now[new_ind_now]
labels_before_new = labels_before[new_ind_bef]

boolar_new = np.equal(labels_before_new, labels_now_new)
num_no_na = boolarr.sum()
frac_no_na = num_no_na / len(labels_before_new)
print(frac_no_na)

# excluding the N/A labels gives a self test accuracy of 75%
# so we need Harold to perform 52% valid accuracy to be at 70% of my own accuracy
#now let's make a confusion matrix with the 3 labels that only looks at validation set (first 5 perc)
