"""File that generate a confusion matrix from given checkpoint. Specify train/valid/test"""
import os
import sys
sys.path.append(os. path. abspath('..'))

import default_config as config
import datasets
from models import dgm_lstm_vae

import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

model = dgm_lstm_vae.DeepGenerativeModel(
    n_layers=config.n_layers,
    input_dim=config.input_dim,
    h_dim=config.h_dim,
    latent_dim=config.latent_dim,
    output_dim=config.input_dim,
    seq_len=config.seq_len,
    neg_slope=config.neg_slope,
    label_dim=config.label_dim,
    batch_size=config.batch_size,
    h_dim_classif=config.h_dim_classif,
    neg_slope_classif=config.neg_slope_classif,
    n_layers_classif=config.n_layers_classif,
    bias=None,
    batch_norm=True,
    classifier=config.classifier
).to(config.device)

labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, \
    labels_valid, labelled_data_test, labels_test, unlabelled_data_test = \
    datasets.get_dgm_data(config)

old_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), "saved/" + config.load_from_checkpoint + ".pt")
checkpoint = torch.load(old_checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])
latest_epoch = checkpoint['epoch']


purpose = 'test' #valid, test

if purpose == 'train':
    x = torch.from_numpy(labelled_data_train.dataset)
    y = torch.squeeze(torch.from_numpy(labels_train.dataset))

if purpose == 'valid':
    x = torch.from_numpy(labelled_data_valid.dataset)
    y = torch.squeeze(torch.from_numpy(labels_valid.dataset))

if purpose == 'test':
    x = torch.from_numpy(labelled_data_test.dataset)
    y = torch.squeeze(torch.from_numpy(labels_test.dataset))

x = x.to(config.device)

logits = model.classify(x)

y_pred = (torch.max(logits, 1).indices).float()

conf_mat = confusion_matrix(
    y.cpu().detach().numpy(), 
    y_pred.cpu().detach().numpy(),
    normalize = 'true'
    )

classes = ['Low', 'Medium', 'High']
accuracies = conf_mat/conf_mat.sum(1)

plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size':'13'})
fig, ax = plt.subplots(figsize=(3,3))
fig.set_figheight(6)
fig.set_figwidth(6)

cnorm = matplotlib.colors.Normalize(vmin=0, vmax=1)

cb = ax.imshow(accuracies, cmap='Blues', norm=cnorm)
plt.xticks(range(len(classes)), classes,rotation=0)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        color='black' if accuracies[j,i] < 0.5 else 'white'
        ax.annotate('{:.2f}'.format(conf_mat[j,i]), (i,j), 
                    color=color, va='center', ha='center')

plt.colorbar(cb, ax=ax, shrink=0.81)
plt.title('Labanet\'s confusion matrix \n On ' + purpose + ' dataset')
plt.ylabel('Ground truth')
plt.xlabel('LabaNet predicts')
plt.savefig(fname="evaluate/confusion/conf_labanet_" + purpose + ".png", dpi=1200)