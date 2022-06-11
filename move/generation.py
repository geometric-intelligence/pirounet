"""File that generate a confusion matrix from given checkpoint. Specify train/valid/test"""
import os
import sys
#sys.path.append(os. path. abspath('..'))
from os.path import exists
import default_config as config
import datasets
from models import dgm_lstm_vae

import torch
from torch.autograd import Variable

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

# 1. change default_config to load desired checkpoit.
# 2. make default_config match classifier, h_dim, h_dim_class, batch_size
# 3. pick empty device
####################################################


for label in range(config.label_dim):
    filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "evaluate/generate/" + config.run_name + '_lab' + str(label))

    if exists(filepath_for_artifacts) is False:
        os.mkdir(filepath_for_artifacts)

    for i in range(30):
        generate_f.generate_and_save(
            model=model, 
            purpose=None,
            epoch=latest_epoch + i + 1, 
            y_given=label, 
            config=config,
            log_to_wandb=False,
            single_epoch=filepath_for_artifacts
        )