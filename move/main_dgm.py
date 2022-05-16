"""Main file performing training with labels (semi-supervised)."""

import logging

logging.basicConfig(level=logging.INFO)

import os
import sys
import warnings

import default_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

import datasets
import generate
import nn
import torch
import train
import train_dgm

import wandb

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
        "h_features_loop": default_config.h_features_loop,
        "latent_dim": default_config.latent_dim,
        "label_features": default_config.label_features,
    },
)


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
    input_features=3 * 53,
    h_features_loop=default_config.h_features_loop,
    latent_dim=default_config.latent_dim,
    output_features=3 * 53,
    seq_len=default_config.seq_len,
    negative_slope=default_config.negative_slope,
    label_features=default_config.label_features,
).to(DEVICE)

labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, \
    labels_valid, labelled_data_test, labels_test, unlabelled_data_test = \
    datasets.get_dgm_data(default_config)


wandb.watch(model, train.get_loss, log="all", log_freq=100)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

train_dgm.run_train_dgm(
    model,
    labelled_data_train,
    labels_train,
    unlabelled_data_train,
    labelled_data_valid,
    labels_valid,
    labelled_data_test,
    labels_test,
    unlabelled_data_test,
    optimizer,
    default_config.epochs,
    default_config.label_features,
    checkpoint=False
)

# generate
artifact_maker = generate.Artifact(model)
for label in range(1, default_config.label_features + 1):
    artifact_maker.generatecond(y_given=label)

wandb.finish()
