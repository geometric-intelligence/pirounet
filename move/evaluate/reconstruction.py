"""File that generate a confusion matrix from given checkpoint. Specify train/valid/test"""
import os
import sys

# sys.path.append(os. path. abspath('..'))
from os.path import exists

import datasets
import default_config as config
import models.utils as utils
import numpy as np
import torch
from evaluate import generate_f
from models import dgm_lstm_vae
from torch.autograd import Variable

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
    classifier=config.classifier,
).to(config.device)

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

old_checkpoint_filepath = os.path.join(
    os.path.abspath(os.getcwd()), "saved/" + config.load_from_checkpoint + ".pt"
)
checkpoint = torch.load(old_checkpoint_filepath)
model.load_state_dict(checkpoint["model_state_dict"])
latest_epoch = checkpoint["epoch"]

# 1. change default_config to load desired checkpoit.
# 2. make default_config match classifier, h_dim, h_dim_class, batch_size
# 3. pick empty device
# 4. select purpose of this reconstruction
# if purpose is test: change batch_size to 1
####################################################

purpose = "train_qty"  # valid or test, valid_qty, test_qty
####################################################

filepath_for_artifacts = os.path.join(
    os.path.abspath(os.getcwd()),
    "evaluate/reconstruct/" + config.run_name + "_" + purpose,
)

if exists(filepath_for_artifacts) is False:
    os.mkdir(filepath_for_artifacts)

batch = 0
onehot_encoder = utils.make_onehot_encoder(3)
if purpose == "valid":
    for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):

        batch += 1

        x, y = Variable(x), Variable(y)
        x, y = x.to(config.device), y.to(config.device)

        batch_one_hot = utils.batch_one_hot(y, config.label_dim)
        y = batch_one_hot.to(config.device)

        generate_f.reconstruct(
            model=model,
            epoch=latest_epoch,
            input_data=x,
            input_label=y,
            purpose="valid_{}".format(batch),
            config=config,
            log_to_wandb=False,
            single_epoch=filepath_for_artifacts,
            comic=True,
        )


if purpose == "test":  # change batch_size to 1
    for i_batch, (x, y) in enumerate(zip(labelled_data_test, labels_test)):

        batch += 1

        x, y = Variable(x), Variable(y)
        x, y = x.to(config.device), y.to(config.device)

        batch_one_hot = torch.zeros((1, 1, config.label_dim))
        for y_i in y:
            y_i_enc = onehot_encoder(y_i.item())
            y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
            batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)
        batch_one_hot = batch_one_hot[1:, :, :]
        y = batch_one_hot.to(config.device)

        generate_f.reconstruct(
            model=model,
            epoch=latest_epoch,
            input_data=x,
            input_label=y,
            purpose="test_{}".format(batch),
            config=config,
            log_to_wandb=False,
            single_epoch=filepath_for_artifacts,
            comic=True,
        )


if purpose == "test_qty":  # change batch_size to 1
    D_total = 0
    batches_seen = 0
    for i_batch, (x, y) in enumerate(zip(labelled_data_test, labels_test)):
        x, y = Variable(x), Variable(y)
        x, y = x.to(config.device), y.to(config.device)

        batch_one_hot = torch.zeros((1, 1, config.label_dim))
        for y_i in y:
            y_i_enc = onehot_encoder(y_i.item())
            y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
            batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)
        batch_one_hot = batch_one_hot[1:, :, :]
        y = batch_one_hot.to(config.device)

        x_recon = model(x, y)  # has shape [batch_size, seq_len, 159]
        # sum over all 159 coordinates and

        D_this_batch = 0

        for i in range(len(x)):
            x_seq = x[i]  # pick one sequence in batch
            x_seq = x_seq.reshape((config.seq_len, -1, 3)).cpu().data.numpy()
            x_recon_seq = x_recon[i]
            x_recon_seq = x_recon_seq.reshape((config.seq_len, -1, 3))
            x_recon_seq = x_recon_seq.cpu().data.numpy()

            d = (x_seq - x_recon_seq) ** 2
            d = np.sum(d, axis=2)
            d = np.sqrt(d)  # shape [40,53]
            d_pose = np.sum(d, axis=1)

            D = d.reshape(-1)
            D = np.mean(D)
            # D = np.sum(np.sum(d, axis = 1), axis=0) # shape [1,] is D for 1 sequence
            D_this_batch += D  # make D for all sequences in batch, [80,]

        # D for all batches
        batches_seen += 1
        D_total += D_this_batch
    # average D for test dataset

    D_test = D_total / (batches_seen * 1)
    print(f"the average D over the test data ({batches_seen} batches) is {D_test}")

if purpose == "valid_qty":  # change batch_size to 1
    D_total = 0
    batches_seen = 0

    for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):
        x, y = Variable(x), Variable(y)
        x, y = x.to(config.device), y.to(config.device)

        batch_one_hot = torch.zeros((1, 1, config.label_dim))
        for y_i in y:
            y_i_enc = onehot_encoder(y_i.item())
            y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
            batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)
        batch_one_hot = batch_one_hot[1:, :, :]
        y = batch_one_hot.to(config.device)

        x_recon = model(x, y)  # has shape [batch_size, seq_len, 159]
        # sum over all 159 coordinates and

        D_this_batch = 0

        for i in range(len(x)):
            x_seq = x[i]  # pick one sequence in batch
            x_seq = x_seq.reshape((config.seq_len, -1, 3)).cpu().data.numpy()
            x_recon_seq = x_recon[i]
            x_recon_seq = x_recon_seq.reshape((config.seq_len, -1, 3))
            x_recon_seq = x_recon_seq.cpu().data.numpy()

            d = (x_seq - x_recon_seq) ** 2
            d = np.sum(d, axis=2)
            d = np.sqrt(d)  # shape [40,53]
            d_pose = np.sum(d, axis=1)

            D = d.reshape(-1)
            D = np.mean(D)
            # D = np.sum(np.sum(d, axis = 1), axis=0) # shape [1,] is D for 1 sequence
            D_this_batch += D  # make D for all sequences in batch, [80,]

        # D for all batches
        batches_seen += 1
        D_total += D_this_batch
    # average D for valid dataset

    D_valid = D_total / (batches_seen * config.batch_size)
    print(f"the average D over the valid data ({batches_seen} batches) is {D_valid}")


if purpose == "train_qty":  # change batch_size to 1
    D_total = 0
    batches_seen = 0

    for i_batch, (x, y) in enumerate(zip(labelled_data_train, labels_train)):
        x, y = Variable(x), Variable(y)
        x, y = x.to(config.device), y.to(config.device)

        batch_one_hot = torch.zeros((1, 1, config.label_dim))
        for y_i in y:
            y_i_enc = onehot_encoder(y_i.item())
            y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
            batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)
        batch_one_hot = batch_one_hot[1:, :, :]
        y = batch_one_hot.to(config.device)

        x_recon = model(x, y)  # has shape [batch_size, seq_len, 159]
        # sum over all 159 coordinates and

        D_this_batch = 0

        for i in range(len(x)):
            x_seq = x[i]  # pick one sequence in batch
            x_seq = x_seq.reshape((config.seq_len, -1, 3)).cpu().data.numpy()
            x_recon_seq = x_recon[i]
            x_recon_seq = x_recon_seq.reshape((config.seq_len, -1, 3))
            x_recon_seq = x_recon_seq.cpu().data.numpy()

            d = (x_seq - x_recon_seq) ** 2
            d = np.sum(d, axis=2)
            d = np.sqrt(d)  # shape [40,53]
            d_pose = np.sum(d, axis=1)

            D = d.reshape(-1)
            D = np.mean(D)
            # D = np.sum(np.sum(d, axis = 1), axis=0) # shape [1,] is D for 1 sequence
            D_this_batch += D  # make D for all sequences in batch, [80,]

        # D for all batches
        batches_seen += 1
        D_total += D_this_batch
    # average D for valid dataset

    D_valid = D_total / (batches_seen * config.batch_size)
    print(f"the average D over the train data ({batches_seen} batches) is {D_valid}")
