import os
import sys
import warnings
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import wandb
import logging

from nn import *
from config import *

logging.info('Initialize WandB project')


wandb.init(project="lstm_vae_spring", entity="mathildepapillon")
wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size": batch_size,
  "seq_len": seq_len
}

logging.info('Run server specific commands')    
SERVER = 'pod' #colab
if SERVER == 'colab':
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd /content/drive/MyDrive/colab-github/move/dance
    sys.path.append(os.path.dirname(os.getcwd()))
    warnings.filterwarnings('ignore')
    # %load_ext autoreload
    # %autoreload 2

if SERVER == 'pod':
    sys.path.append(os.path.dirname(os.getcwd()))
    warnings.filterwarnings('ignore')



logging.info('Initialize model')
model = LstmVAE(n_layers=2, 
        input_features=3 * 53, 
        h_features_loop=32, 
        latent_dim=32, 
        kl_weight=0,
        output_features=3 * 53,
        seq_len=128,
        negative_slope=0.2).to(device) 

pytorch_total_params = sum(p.numel() for p in model.parameters())


logging.info('Load data')
ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data()
my_data = ds_all_centered.reshape((ds_all.shape[0], -1))

logging.info('Make seq_data of shape [number of seq, seq_len, input_features]')
seq_data = np.zeros((my_data.shape[0]-seq_len, seq_len, my_data.shape[1]))
for i in range((ds_all.shape[0]-seq_len)):
    seq_data[i] = my_data[i:i+seq_len]

logging.info('Make training data and validing data')
five_perc = int(round(seq_data.shape[0]*0.05))
ninety_perc = seq_data.shape[0] - (2*five_perc)
valid_ds = seq_data[:five_perc, :, :]
test_ds = seq_data[five_perc:(2*five_perc), :, :]
train_ds = seq_data[ninety_perc:, :, :]

logging.info('Make torch tensor in batches')
data_train_torch = torch.utils.data.DataLoader(train_ds, 
    batch_size=batch_size, num_workers=2)
data_valid_torch = torch.utils.data.DataLoader(valid_ds, 
    batch_size=batch_size, num_workers=2)
data_test_torch = torch.utils.data.DataLoader(test_ds, 
    batch_size=1, num_workers=2)


logging.info('Train/validate and record loss')
wandb.watch(model, get_loss, log="all", log_freq=100)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
run_train(model, data_train_torch, data_valid_torch, data_test_torch, get_loss, optimizer, epochs)

wandb.finish()