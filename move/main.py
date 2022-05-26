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
import evaluate.generate_f as generate_f
import models.dgm_lstm_vae as dgm_lstm_vae
import torch
import train
import wandb

logging.info(f"Using PyTorch version: {torch. __version__}")

# Can be replaced by logging.DEBUG or logging.WARNING
warnings.filterwarnings("ignore")

# The config put in wandb.init is treated as default:
# - it is filled with values from the default_config.py file
# - it will be overwritten by any sweep
# After wandb is initialize, use wandb.config (and not default_config)
# to get the config parameters of the run, potentially coming from a sweep.
wandb.init(
    project="move_labelled_nina",
    entity="bioshape-lab",
    config={
        "run_name": default_config.run_name,
        "load_from_checkpoint": default_config.load_from_checkpoint,
        "epochs": default_config.epochs,
        "learning_rate": default_config.learning_rate,
        "batch_size": default_config.batch_size,
        "seq_len": default_config.seq_len,
        "with_clip": default_config.with_clip,
        "input_dim": default_config.input_dim,
        "kl_weight": default_config.kl_weight,
        "neg_slope": default_config.neg_slope,
        "n_layers": default_config.n_layers,
        "h_dim": default_config.h_dim,
        "latent_dim": default_config.latent_dim,
        "neg_slope_classif": default_config.neg_slope_classif,
        "n_layers_classif": default_config.n_layers_classif,
        "h_dim_classif": default_config.h_dim_classif,
        "label_dim": default_config.label_dim,
        "device": default_config.device,
    },
)
config = wandb.config
logging.info(f"Config: {config}")
logging.info(f"---> Using device {config.device}")

wandb.run.name = default_config.run_name


logging.info("Initialize model")
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
).to(config.device)
# encoder = transformer.Encoder()
# decoder = transformer.Decoder()
# model = transformer.CVAE()

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
train.run_train_dgm(
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
    config=config,
)

# generate
for label in range(config.label_dim):
    generate_f.generate(model, y_given=label, config=config)

wandb.finish()
