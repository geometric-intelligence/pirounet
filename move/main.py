"""Main file performing training."""

import logging
import os
import sys
import warnings

import datasets
import default_config
import nn
import torch
import train
import wandb


# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

logging.info(f"PyTorch version: {torch. __version__}")

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
logging.info(f"Using device {DEVICE}")

# The config put in wandb.init is treated as default
# and would be overwritten by a sweep
wandb.init(
    project="move",
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
    },
)
config = wandb.config
logging.info("Config: {config}")

logging.info("Run server specific commands")
SERVER = "pod"  # colab
if SERVER == "colab":
    from google.colab import drive

    drive.mount("/content/drive")
    # %cd /content/drive/MyDrive/colab-github/move/dance
    sys.path.append(os.path.dirname(os.getcwd()))

elif SERVER == "pod":
    sys.path.append(os.path.dirname(os.getcwd()))


logging.info("Initialize model")
model = nn.LstmVAE(
    n_layers=config.n_layers,
    input_features=3 * 53,
    h_features_loop=config.h_features_loop,
    latent_dim=config.latent_dim,
    kl_weight=config.kl_weight,
    output_features=3 * 53,
    seq_len=config.seq_len,
    negative_slope=config.negative_slope,
).to(DEVICE)

data_train_torch, data_valid_torch, data_test_torch = datasets.get_mariel_data(config)

wandb.watch(model, train.get_loss, log="all", log_freq=100)
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
)

train.run_train(
    model,
    data_train_torch,
    data_valid_torch,
    data_test_torch,
    train.get_loss,
    optimizer,
    config.epochs,
)

wandb.finish()
