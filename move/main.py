"""Main file performing training with labels (semi-supervised)."""

import logging

logging.basicConfig(level=logging.INFO)

import os
import warnings

import datasets
import default_config
import evaluate.generate_f as generate_f
import models.dgm_lstm_vae as dgm_lstm_vae
import torch
import train

import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

logging.info(f"Using PyTorch version: {torch. __version__}")

warnings.filterwarnings("ignore")

wandb.init(
    project=default_config.project,
    entity=default_config.entity,
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
        "effort": default_config.effort,
        # "fraction_label": default_config.fraction_label,
        "train_ratio": default_config.train_ratio,
        "train_lab_frac": default_config.train_lab_frac,
    },
)

config = wandb.config
# wandb.run.name = default_config.run_name

logging.info(f"Config: {config}")
logging.info(f"---> Using device {config.device}")
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
) = datasets.get_model_data(config)

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
    optimizer,
    config=config,
)

wandb.finish()
