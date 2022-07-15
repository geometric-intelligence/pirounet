"""File that generate a confusion matrix from given checkpoint. Specify train/valid/test"""

import os
import sys

sys.path.append(os.path.abspath(".."))

import datasets
import default_config as config
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models import dgm_lstm_vae
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    multilabel_confusion_matrix,
)

# only compare elements that are the same. get array of true/false for each label, and in total

labels_before, labels_before_ind = datasets.load_labels(
    effort="time",
    filepath="/home/papillon/move/move/data/redid_labels_from_app.csv",
    no_NA=True,
    augment=True,
)

labels_now, labels_now_ind = datasets.load_labels(
    effort="time",
    filepath="/home/papillon/move/move/data/labels_from_app.csv",
    no_NA=True,
    augment=True,
)

labels_before = labels_before - 1.0
labels_now = labels_now - 1.0

one_perc_lab = int(round(len(labels_now) * 0.01))
five_perc_lab = int(one_perc_lab * 3)


purpose = "test"

if purpose == "test":
    labels_now = labels_now[
        ((five_perc_lab * 19) + (one_perc_lab * 3)) :,
    ][:, 0]
    print(labels_now.shape)
    labels_now_ind = labels_now_ind[
        ((five_perc_lab * 19) + (one_perc_lab * 3)) :,
    ][:, 0]

if purpose == "valid":
    labels_now = labels_now[
        :five_perc_lab,
    ][:, 0]
    labels_now_ind = labels_now_ind[
        :five_perc_lab,
    ][:, 0]

if purpose == "all":
    labels_now = labels_now[:, 0]
    labels_now_ind = labels_now_ind[:, 0]

# find the matching labels in the second labelled set.
# We want to match labels from the first 5% o the augmented set Harold uses
res = [i for i in labels_now_ind if i in labels_before_ind]
new_ind_now = []
new_ind_bef = []

for i in range(len(res)):
    ind = np.where(res[i] == labels_now_ind)
    new_ind_now.append(ind)
    ind_bef = np.where(res[i] == labels_before_ind)
    new_ind_bef.append(ind_bef)

new_ind_now = np.array(new_ind_now)[:, 0, 0]
new_ind_bef = np.array(new_ind_bef)[:, 0, 0]

v_labels_now_new = labels_now[new_ind_now]  # [:,0]
v_labels_before_new = labels_before[new_ind_bef][:, 0]

conf_mat = confusion_matrix(v_labels_now_new, v_labels_before_new, normalize="true")
classes = ["Low", "Medium", "High"]
accuracies = conf_mat / conf_mat.sum(1)

plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.size": "13"})
fig, ax = plt.subplots(figsize=(3, 3))
fig.set_figheight(6)
fig.set_figwidth(6)

cnorm = matplotlib.colors.Normalize(vmin=0, vmax=1)

cb = ax.imshow(accuracies, cmap="Blues", norm=cnorm)
plt.xticks(range(len(classes)), classes, rotation=0)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        color = "black" if accuracies[j, i] < 0.5 else "white"
        ax.annotate(
            "{:.2f}".format(conf_mat[j, i]),
            (i, j),
            color=color,
            va="center",
            ha="center",
        )

plt.colorbar(cb, ax=ax, shrink=0.81)
plt.xlabel("Labeler re-predicts")
plt.ylabel("Ground truth")
plt.title("Labeler's confusion matrix \n On comparable " + str(purpose) + " dataset")
plt.savefig(fname="evaluate/confusion/conf_self_" + str(purpose) + ".png", dpi=1200)
