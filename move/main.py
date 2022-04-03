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

logging.info("Setup device")
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


logging.info("Initialize WandB project.")

wandb.init(
    project="move",
    entity="ninamiolane",
    config={
        "learning_rate": default_config.learning_rate,
        "epochs": default_config.epochs,
        "batch_size": default_config.batch_size,
        "seq_len": default_config.seq_len,
    },
)
config = wandb.config

logging.info(f"Config: {config}")


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
    n_layers=2,
    input_features=3 * 53,
    h_features_loop=32,
    latent_dim=32,
    kl_weight=0,
    output_features=3 * 53,
    seq_len=config.seq_len,
    negative_slope=0.2,
).to(DEVICE)

logging.info("Load data")
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
