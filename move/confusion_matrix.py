"""Main file performing training with labels (semi-supervised)."""

import logging
from random import triangular

logging.basicConfig(level=logging.INFO)

import os
import sys
import warnings

import default_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

import datasets
import nn
import generate_f
import utils


import wandb
import torch
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

logging.info('TORCH')
logging.info(torch. __version__)

# Can be replaced by logging.DEBUG or logging.WARNING
warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
logging.info(f"Using device {DEVICE}")

# The config put in the init is treated as default
# and would be overwritten by a sweep
wandb.init(
    project="move_labelled",
    entity="bioshape-lab",
    config={
        "learning_rate": default_config.learning_rate,
        "epochs": default_config.epochs,
        "batch_size": default_config.batch_size,
        "seq_len": default_config.seq_len,
        "kl_weight": default_config.kl_weight,
        "negative_slope": default_config.negative_slope,
        "n_layers": default_config.n_layers,
        "h_dim": default_config.h_dim,
        "latent_dim": default_config.latent_dim,
        "label_features": default_config.label_features,
        "neg_slope_classif": default_config.neg_slope_classif,
        "n_layers_classif": default_config.n_layers_classif,
        "h_dim_classif": default_config.h_dim_classif,
    },
)
wandb.run.name = default_config.run_name


logging.info("Run server specific commands")
SERVER = "pod"  # colab
if SERVER == "colab":
    from google.colab import drive

    drive.mount("/content/drive")
    # %cd /content/drive/MyDrive/colab-github/move/dance
    syspath.append(os.path.dirname(os.getcwd()))

elif SERVER == "pod":
    sys.path.append(os.path.dirname(os.getcwd()))

logging.info("Initialize model")
model = nn.DeepGenerativeModel(
    n_layers=default_config.n_layers,
    input_features=default_config.input_features,
    h_dim=default_config.h_dim,
    latent_dim=default_config.latent_dim,
    output_features=3 * 53,
    seq_len=default_config.seq_len,
    negative_slope=default_config.negative_slope,
    label_features=default_config.label_features,
    batch_size=default_config.batch_size,
    h_dim_classif=default_config.h_dim_classif,
    neg_slope_classif=default_config.neg_slope_classif,
    n_layers_classif=default_config.n_layers_classif,
    bias=None,
    batch_norm=True
).to(DEVICE)

labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, \
    labels_valid, labelled_data_test, labels_test, unlabelled_data_test = \
    datasets.get_dgm_data(default_config)

old_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), "saved/" + default_config.load_from_checkpoint)
checkpoint = torch.load(old_checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])
latest_epoch = checkpoint['epoch']

onehot_encoder = utils.make_onehot_encoder(default_config.label_features)

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

x = x.to(DEVICE)

logits = model.classify(x)

y_pred = (torch.max(logits, 1).indices).float()

conf_mat = ConfusionMatrixDisplay.from_predictions(
    y.cpu().detach().numpy(), 
    y_pred.cpu().detach().numpy(),
    #labels = ['Low', 'Medium', 'High', 'N/A'],
    #cmap = 'Blues'
    )

plt.savefig(fname="saved/confusion/conf_" + str(purpose) + "_" + 
        default_config.run_name + "_" + str(latest_epoch) + ".png")

wandb.finish()

