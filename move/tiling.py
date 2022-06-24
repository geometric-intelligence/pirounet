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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
############################################################
purpose = 'train'
step_size = 0.2
dances_per_tile = [7,7,1]
density_thresh = 0.9
############################################################

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

z_0 = np.squeeze(np.array(z_0))
z_1 = np.squeeze(np.array(z_1))
z_2 = np.squeeze(np.array(z_2))

z_0_center = np.mean(z_0, axis = 0)
z_1_center = np.mean(z_1, axis = 0)
z_2_center = np.mean(z_2, axis = 0)

pca0 = PCA(n_components=2).fit(z_0 - z_0_center)
z_0_transf = pca0.transform(z_0 - z_0_center)

pca1 = PCA(n_components=2).fit(z_1 - z_1_center)
z_1_transf = pca1.transform(z_1 - z_1_center)

pca2 = PCA(n_components=2).fit(z_2 - z_2_center)
z_2_transf = pca2.transform(z_2 - z_2_center)

all_z = [z_0_transf, z_1_transf, z_2_transf]

z_min, z_max = -8, 8

grid_xs = np.arange(z_min, z_max, step_size)
grid_ys = - np.arange(z_min, z_max, step_size)
n_xs = len(grid_xs)
n_ys = len(grid_ys)


count = np.zeros((config.label_dim, n_xs, n_ys))

for i, y_coord in enumerate(grid_ys):
    for j, x_coord in enumerate(grid_xs):
        for y in range(config.label_dim):
            for z in all_z[y]:
                if x_coord < z[0] and z[0] < (x_coord + step_size):
                    if y_coord < z[1] and z[1] < (y_coord + step_size):
                        count[y, i, j] += 1


sum_of_counts = np.sum(count, axis=0)
density_0 = count[0] / sum_of_counts
density_1 = count[1] / sum_of_counts
density_2 = count[2] / sum_of_counts
density = [count[i] / sum_of_counts for i in range(config.label_dim)]

# create array of high density tiles that we can sample from later
high_0 = []
high_1 = []
high_2 = []
all_high = []
for y in range(config.label_dim):
    high = []
    for i, y_coord in enumerate(grid_ys):
        for j, x_coord in enumerate(grid_xs):
            if density[y][i, j] > density_thresh and count[y, i, j] > dances_per_tile[y]:
                high.append((x_coord,y_coord))             
            # if density_0[i, j] > density_thresh and count[0, i, j] > dances_per_tile:
            #     high_0.append((x_coord,y_coord)) 
            # if density_1[i, j] > density_thresh and count[1, i, j] > dances_per_tile:
            #     high_1.append((x_coord,y_coord))
            # if density_2[i, j] > density_thresh and count[2, i, j] > 1:
            #     high_2.append((x_coord,y_coord))
    all_high.append(high)

print('amount hitting threshold')
print(len(all_high[0]))
print(len(all_high[1]))
print(len(all_high[2]))

fig, axs = plt.subplots(1,3)
axs[0].imshow(density[0])
axs[1].imshow(density[1])
axs[2].imshow(density[2])
filepath_for_density = os.path.join(os.path.abspath(os.getcwd()), "evaluate/z_tiling/" + config.load_from_checkpoint)
if exists(filepath_for_density) is False:
    os.mkdir(filepath_for_density)
plt.savefig(f'{filepath_for_density}/tiling_dens_count_{step_size}_{purpose}.png')

n_test = 30
for y in range(config.label_dim):
    for i in range(n_test):
        # decode within high density tile
        tile_to_pick = np.random.randint(0, len(all_high[y]))
        tile = all_high[y][tile_to_pick]
        zx = np.random.uniform(tile[0], tile[0] + step_size)
        zy = np.random.uniform(tile[1], tile[1] + step_size)
        z = np.array((zx, zy))
        z = pca0.inverse_transform(z) + z_0_center

        # back transform with PCA

        z_within_tile = torch.tensor(z).reshape(1, -1).to(config.device).float()

        # while torch.dist(z_within_radius, zs[y]).to(config.device) > torch.linalg.norm(zstd_s[y]).to(config.device):
        #     continue

        x_create = model.sample(z_within_tile, onehot_encoder(0).reshape((1, 3)).to(config.device))
        x_create_formatted = x_create[0].reshape((config.seq_len, -1, 3))

        filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "evaluate/z_tiling/" + config.load_from_checkpoint)
        if exists(filepath_for_artifacts) is False:
            os.mkdir(filepath_for_artifacts)

        filepath_for_label = os.path.join(os.path.abspath(os.getcwd()), "evaluate/z_tiling/" + config.load_from_checkpoint + "/label" + str(y))
        if exists(filepath_for_label) is False:
            os.mkdir(filepath_for_label)

        name = f"{i}_tile_{y}_dance.gif"

        fname0 = generate_f.animatestick(
            x_create_formatted,
            fname=os.path.join(str(filepath_for_label), name),
            ghost=None,
            dot_alpha=0.7,
            ghost_shift=0.2,
            condition=0,
        )


# H0, grid_xs0, grid_ys0 = np.histogram2d(z_0_transf[0], z_0_transf[1], bins=(grid_xs, grid_ys), density=True)
# H0 = H0.T
# print(H0)
# H1, grid_xs1, grid_ys1 = np.histogram2d(z_1_transf[0], z_1_transf[1], bins=(grid_xs, grid_ys), density=True)
# H1 = H1.T
# print(H1)
# H2, grid_xs2, grid_ys2 = np.histogram2d(z_2_transf[0], z_2_transf[1], bins=(grid_xs, grid_ys), density=True)
# H2 = H2.T
# print(H2)
# fig, axs = plt.subplots(1,3)
# im0 = axs[0].imshow(H0, interpolation='nearest', origin='lower',
#     extent=[grid_xs0[0], grid_xs0[-1], grid_ys0[0], grid_ys0[-1]])
# cbar = plt.colorbar(im0)

# im1 = axs[1].imshow(H1, interpolation='nearest', origin='lower',
#     extent=[grid_xs1[0], grid_xs1[-1], grid_ys1[0], grid_ys1[-1]])
# cbar = plt.colorbar(im1)

# im2 = axs[2].imshow(H2, interpolation='nearest', origin='lower',
#     extent=[grid_xs2[0], grid_xs2[-1], grid_ys2[0], grid_ys2[-1]])
# cbar = plt.colorbar(im2)

# filepath_for_density = os.path.join(os.path.abspath(os.getcwd()), "evaluate/z_tiling/" + config.load_from_checkpoint)
# if exists(filepath_for_density) is False:
#     os.mkdir(filepath_for_density)
# plt.savefig(f'{filepath_for_density}/tiling_density_{step_size}.png')