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
    label_type = 'time',
    filepath="/home/papillon/move/move/data/manual_labels.csv",
    no_NA=False,
    augment = False)

labels_now, labels_now_ind = datasets.load_labels(
    label_type = 'time',
    filepath="/home/papillon/move/move/data/labels_mathilde.csv",
    no_NA=False,
    augment = False)


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

boolarr = np.equal(labels_before, labels_now)
num = boolarr.sum()
frac = num / len(labels_before)
print(frac)
