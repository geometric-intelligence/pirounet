"""File that generates a confusion matrix from given checkpoint. Specify train/valid/test"""
import os
import sys
#sys.path.append(os. path. abspath('..'))
from os.path import exists
import default_config as config
import datasets
from models import dgm_lstm_vae

import torch
from torch.autograd import Variable
import numpy as np
import csv

from evaluate import generate_f
import models.utils as utils

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
    classifier=config.classifier
).to(config.device)

labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, \
    labels_valid, labelled_data_test, labels_test, unlabelled_data_test = \
    datasets.get_dgm_data(config)

old_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), "saved/" + config.load_from_checkpoint + ".pt")
checkpoint = torch.load(old_checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])
latest_epoch = checkpoint['epoch']

purpose = 'test'
if purpose == 'valid':
    labelled_data = labelled_data_valid
    labels = labels_valid
if purpose == 'train':
    labelled_data = labelled_data_train
    labels = labels_train
if purpose == 'test':
    labelled_data = labelled_data_test
    labels = labels_test
# encode training data and see where the average latent variable is situated for each effort
z_0 = []
z_1 = []
z_2 = []
batch = 0
onehot_encoder = utils.make_onehot_encoder(config.label_dim)
for i_batch, (x, y) in enumerate(zip(labelled_data, labels)):

    batch +=1 

    x, y = Variable(x), Variable(y)
    x, y = x.to(config.device), y.to(config.device)

    y_label = y.item()
    batch_one_hot = utils.batch_one_hot(y, config.label_dim)
    y = batch_one_hot.to(config.device)

    z, z_mu, z_logvar = model.encode(x, y)

    if y_label == 0:
        z_0.append(z.cpu().data.numpy())
    if y_label == 1:
        z_1.append(z.cpu().data.numpy())
    if y_label == 2:
        z_2.append(z.cpu().data.numpy())

z_0 = np.array(z_0)
z_1 = np.array(z_1)
z_2 = np.array(z_2)

z_0_mean = torch.tensor(np.mean(z_0, axis = 0)).to(config.device)
z_1_mean = torch.tensor(np.mean(z_1, axis = 0)).to(config.device)
z_2_mean = torch.tensor(np.mean(z_2, axis = 0)).to(config.device)
z_0_std = torch.tensor(np.std(z_0, axis = 0)).to(config.device)
z_1_std = torch.tensor(np.std(z_1, axis = 0)).to(config.device)
z_2_std = torch.tensor(np.std(z_2, axis = 0)).to(config.device)
print(z_0.shape)
print(z_0_mean.shape)
print(z_0_std.shape)

zs = torch.stack((z_0_mean, z_1_mean, z_2_mean)).to(config.device)
zstd_s = torch.stack((z_0_std, z_1_std, z_2_std)).to(config.device)

dist_01 = torch.dist(z_0_mean, z_1_mean).to(config.device)
dist_02 = torch.dist(z_0_mean, z_2_mean).to(config.device)
dist_12 = torch.dist(z_1_mean, z_2_mean).to(config.device)

# relative distance between them
print(f'Distance between Low and Medium: {dist_01.item()}')
print(f'Distance between Low and High: {dist_02.item()}')
print(f'Distance between Medium and High: {dist_12.item()}')

# decode each average and generate
for y in range(config.label_dim):
    x_create = model.sample(zs[y], onehot_encoder(y).reshape((1, 3)).to(config.device))
    x_create_formatted = x_create[0].reshape((config.seq_len, -1, 3))

    filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "evaluate/z_mean/" + config.run_name)
    if exists(filepath_for_artifacts) is False:
        os.mkdir(filepath_for_artifacts)

    name = f"mean_dance_{y}_{purpose}.gif"

    fname0 = generate_f.animatestick(
        x_create_formatted,
        fname=os.path.join(str(filepath_for_artifacts), name),
        ghost=None,
        dot_alpha=0.7,
        ghost_shift=0.2,
        condition=y,
    )

# decode within radius away from average
n_radius_randoms = 2
for y in range(config.label_dim):

    for i in range(n_radius_randoms):
        z_within_radius = torch.randn(size=(1, config.latent_dim)).to(config.device)
        while torch.dist(z_within_radius, zs[y]).to(config.device) > torch.linalg.norm(zstd_s[y]).to(config.device):
            continue

        print(f'found z within radius of z{y}_avg')
        
        x_create = model.sample(z_within_radius, onehot_encoder(y).reshape((1, 3)).to(config.device))
        x_create_formatted = x_create[0].reshape((config.seq_len, -1, 3))

        filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "evaluate/z_mean/" + config.run_name)
        if exists(filepath_for_artifacts) is False:
            os.mkdir(filepath_for_artifacts)

        name = f"n{i}_in_rad_{y}_{purpose}.gif"

        fname0 = generate_f.animatestick(
            x_create_formatted,
            fname=os.path.join(str(filepath_for_artifacts), name),
            ghost=None,
            dot_alpha=0.7,
            ghost_shift=0.2,
            condition=y,
        )