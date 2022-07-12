"""Main file performing training with labels (semi-supervised)."""

import logging

logging.basicConfig(level=logging.INFO)

import os
import warnings

import classifier_config as fid_classifier_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = fid_classifier_config.which_device

import datasets
import models.classifiers as classifiers
import fid_train

import torch
import wandb
import numpy as np

logging.info(f"Using PyTorch version: {torch. __version__}")

# Can be replaced by logging.DEBUG or logging.WARNING
warnings.filterwarnings("ignore")

# The config put in wandb.init is treated as default:
# - it is filled with values from the default_config.py file
# - it will be overwritten by any sweep
# After wandb is initialized, use wandb.config (and not default_config)
# to get the config parameters of the run, potentially coming from a sweep.
wandb.init(
    project="classifier_fid",
    entity="bioshape-lab",
    config={
        "run_name": fid_classifier_config.run_name,
        "epochs": fid_classifier_config.epochs,
        "learning_rate": fid_classifier_config.learning_rate,
        "batch_size": fid_classifier_config.batch_size,
        "seq_len": fid_classifier_config.seq_len,
        "input_dim": fid_classifier_config.input_dim,
        "neg_slope_classif": fid_classifier_config.neg_slope_classif,
        "n_layers_class": fid_classifier_config.n_layers_class,
        "h_dim_class": fid_classifier_config.h_dim_class,
        "h_dim": fid_classifier_config.h_dim,
        "n_layers": fid_classifier_config.n_layers,
        "label_dim": fid_classifier_config.label_dim,
        "device": fid_classifier_config.device,
        "effort": fid_classifier_config.effort,
    },
)
config = wandb.config
logging.info(f"Config: {config}")
logging.info(f"---> Using device {config.device}")

#wandb.run.name = fid_classifier_config.run_name

logging.info("Initialize classifier model")
# model = classifiers.FID_lstm_Classifier(
#     input_dim=config.input_dim,
#     h_dim=config.h_dim,
#     h_dim_class=config.h_dim_class,
#     label_dim=config.label_dim,
#     n_layers=config.n_layers,
#     n_layers_class=config.n_layers_class,
#     return_activation=False
# ).to(config.device)
model = classifiers.LinearClassifier(
    input_dim=config.input_dim,
    h_dim=config.h_dim_class,
    label_dim=config.label_dim,
    seq_len=config.seq_len,
    neg_slope=config.neg_slope_classif,
    n_layers=config.n_layers_class,
).to(config.device)

logging.info("Get original data")
(
labelled_data_train, 
labels_train, 
_, 
labelled_data_valid, 
labels_valid, 
labelled_data_test, 
labels_test,
_
) = datasets.get_model_data(config)

# wandb.watch(model, train.get_loss, log="all", log_freq=100)
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
)

logging.info("Train")
fid_train.run_train_classifier(
    model,
    labelled_data_train,
    labels_train,
    labelled_data_valid,
    labels_valid,
    optimizer,
    config=config,
)

wandb.finish()
