"""File that generate a confusion matrix from given checkpoint. Specify train/valid/test"""
import os
import sys
sys.path.append(os. path. abspath('..'))

import default_config as config
import datasets
from models import dgm_lstm_vae

import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

purpose = 'train' #valid, test

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

conf_mat = ConfusionMatrixDisplay.from_predictions(
    y.cpu().detach().numpy(), 
    y_pred.cpu().detach().numpy(),
    #labels = ['Low', 'Medium', 'High', 'N/A'],
    #cmap = 'Blues'
    )

plt.title(str(purpose) + " for " + config.run_name + " at epoch" + str(latest_epoch))
plt.savefig(fname="saved/confusion/conf_" + str(purpose) + "_" + 
        config.run_name + "_" + str(latest_epoch) + ".png")
