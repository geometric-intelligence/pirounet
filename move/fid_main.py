"""Main file performing training with labels (semi-supervised)."""

import logging

logging.basicConfig(level=logging.INFO)

import os
import warnings

import fid_classifier_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

import datasets
import models.classifiers as classifiers
import fid_train

import torch
import wandb

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
        "n_layers_classif": fid_classifier_config.n_layers_classif,
        "h_dim_classif": fid_classifier_config.h_dim_classif,
        "label_dim": fid_classifier_config.label_dim,
        "device": fid_classifier_config.device,
        "classifier": fid_classifier_config.classifier,
        "effort": fid_classifier_config.effort,
    },
)
config = wandb.config
logging.info(f"Config: {config}")
logging.info(f"---> Using device {config.device}")

wandb.run.name = fid_classifier_config.run_name

logging.info("Initialize classifier model")
model = classifiers.FID_Classifier(
    input_dim=config.input_dim,
    h_dim=config.h_dim_classif,
    label_dim=config.label_dim,
    seq_len=config.seq_len,
    neg_slope=config.neg_slope_classif,
    n_layers=config.n_layers_classif,
    return_activation=False
).to(config.device)


logging.info("Get data")
(
    labelled_data_train,
    labels_train,
    unlabelled_data_train,
    labelled_data_valid,
    labels_valid,
    labelled_data_test,
    labels_test,
    unlabelled_data_test,
) = datasets.get_dgm_data(config)

# wandb.watch(model, train.get_loss, log="all", log_freq=100)
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
)

logging.info("Train")
fid_train.run_train_classifier(
    model,
    labelled_data_train,
    labels_train,
    unlabelled_data_train,
    labelled_data_valid,
    labels_valid,
    optimizer,
    config=config,
)

# generate
for label in range(config.label_dim):
    generate_f.generate(model, y_given=label, config=config)

wandb.finish()
