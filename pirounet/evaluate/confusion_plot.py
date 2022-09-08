"""Defines funcitons for generating confusion matrix plots."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def plot_classification_accuracy(model, config, labelled_data, labels, purpose, path):
    """Make and save confusion matrix plot for PirouNet.

    Evaluate classification accuracy for each categorical label, for
    any given dataset.

    Parameters
    ----------
    model :         serialized object
                    Model to evaluate.
    config :        dict
                    Configuration for run.
    labelled_data : array
                    Shape = [n_seqs, seq_len, input_dim]
                    Input sequences to be classified by PirouNet.
    labels :        array
                    Shape = [n_seqs, ]
                    Labels associated to input sequences.
    purpose :       string
                    {"train", "test", "valid"}
                    Nature of input sequences.
    path :          string
                    Path to directory for saving the confusion matrix plot.

    """
    x = torch.from_numpy(labelled_data.dataset)
    y = torch.squeeze(torch.from_numpy(labels.dataset))

    x = x.to(config.device)

    logits = model.classify(x)

    y_pred = (torch.max(logits, 1).indices).float()

    conf_mat = confusion_matrix(
        y.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), normalize="true"
    )

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
    plt.title("PirouNet's confusion matrix \n On " + purpose + " dataset")
    plt.ylabel("Ground truth")
    plt.xlabel("PirouNet predicts")
    plt.savefig(fname=path + f"/confusion_acc_{purpose}.png", dpi=1200)


def plot_recognition_accuracy(human_labels, pirounet_labels, path):
    """Plot recognition accuracy of human labeler via a confusion matrix.

    Measure the classification accuracy of a human labeler with
    respect to PirouNet conditionnally generated sequences.
    all_labels_pirounet has shape = [n_seqs,]
    all_labels_human has shape = [n_seqs, 3], where column 1
    corresponds to sequence index, column 2 to categorical qualitative
    statement (0 = not danceable, 1 = danceable, etc.), and column 3
    to categorical label (what will be compared to PirouNet.)

    In this processing, we get rid of any sequence that was labeled by
    the human with a '4', meaning it was too difficult to classify, or
    deemed "non-applicable".

    Parameters
    ---------
    human_labels :      string
                        Path to csv file containing human-made labels.
    pirounet_labels :   string
                        Path to npy file containing PirouNet labels for
                        same set of sequences.
    path :              string
                        Path to directory for saving confusion matrix plot.
    """
    all_labels_human = np.genfromtxt(open(human_labels), delimiter=",", dtype=None)
    all_labels_pirounet = np.array(np.load(pirounet_labels))

    all_labels_human_no0 = []
    all_labels_pirounet_no0 = []
    for i in range(len(all_labels_human + 1)):
        row_m = all_labels_human[i]
        row_LN = all_labels_pirounet[i]
        if row_m[2] != 4.0:
            all_labels_human_no0.append(row_m)
            all_labels_pirounet_no0.append(row_LN)

    labels_h = np.array(all_labels_human_no0)[:, 2]
    labels_p = np.array(all_labels_pirounet_no0)

    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": "13"})
    conf_mat = confusion_matrix(labels_p, labels_h, normalize="true")
    classes = ["Low", "Medium", "High"]
    accuracies = conf_mat / conf_mat.sum(1)
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.set_figheight(6)
    fig.set_figwidth(7)

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

    plt.colorbar(cb, ax=ax, shrink=0.935)
    plt.xlabel("Labeler blindly predicts")
    plt.ylabel("Neighborhood sampled from")
    plt.title("Labeler versus PirouNet confusion matrix")
    plt.savefig(fname=path + "/recognition_accuracy.png", dpi=1200)
