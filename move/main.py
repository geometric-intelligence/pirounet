import logging
import os
import sys
import warnings

import numpy as np
import torch
import wandb

import default_config
import nn
import train

WANDB = False

logging.info("Initialize WandB project")

if WANDB:
    wandb.init(
        project="new",
        entity="ninamiolane",
        # settings=wandb.Settings(start_method="thread"),
        config={
            "learning_rate": default_config.learning_rate,
            "epochs": default_config.epochs,
            "batch_size": default_config.batch_size,
            "seq_len": default_config.seq_len,
        },
    )
    config = wandb.config

else:
    class Config:
        def __init__(self):
            self.learning_rate = default_config.learning_rate
            self.epochs = default_config.epochs
            self.batch_size = default_config.batch_size
            self.seq_len = default_config.seq_len
    config = Config()

logging.info(f"Config: {config}")


logging.info("Run server specific commands")
SERVER = "pod"  # colab
if SERVER == "colab":
    from google.colab import drive

    drive.mount("/content/drive")
    # %cd /content/drive/MyDrive/colab-github/move/dance
    sys.path.append(os.path.dirname(os.getcwd()))
    warnings.filterwarnings("ignore")

if SERVER == "pod":
    sys.path.append(os.path.dirname(os.getcwd()))
    warnings.filterwarnings("ignore")


logging.info("Initialize model")
model = nn.LstmVAE(
    n_layers=2,
    input_features=3 * 53,
    h_features_loop=32,
    latent_dim=32,
    kl_weight=0,
    output_features=3 * 53,
    seq_len=128,
    negative_slope=0.2,
).to(default_config.device)

logging.info("Load data")
ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = nn.load_data()
my_data = ds_all_centered.reshape((ds_all.shape[0], -1))

logging.info("Make seq_data of shape [number of seq, seq_len, input_features]")
seq_data = np.zeros(
    (my_data.shape[0] - config.seq_len, config.seq_len, my_data.shape[1])
)
for i in range((ds_all.shape[0] - config.seq_len)):
    seq_data[i] = my_data[i : i + config.seq_len]

logging.info("Make training data and validing data")
five_perc = int(round(seq_data.shape[0] * 0.05))
ninety_perc = seq_data.shape[0] - (2 * five_perc)
valid_ds = seq_data[:five_perc, :, :]
test_ds = seq_data[five_perc : (2 * five_perc), :, :]
train_ds = seq_data[ninety_perc:, :, :]

logging.info("Make torch tensor in batches")
data_train_torch = torch.utils.data.DataLoader(
    train_ds, batch_size=config.batch_size, num_workers=2
)
data_valid_torch = torch.utils.data.DataLoader(
    valid_ds, batch_size=config.batch_size, num_workers=2
)
data_test_torch = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=2)


logging.info("Train/validate and record loss")
if WANDB:
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

if WANDB:
    wandb.finish()
