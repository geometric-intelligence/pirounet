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

# 1. change default_config to load desired checkpoint.
# 2. make default_config match classifier, h_dim, h_dim_class, batch_size
# 3. pick empty device
# 4. pick purpose of generation
####################################################
purpose = 'one_move' # artifact, blind, one_move
####################################################

num_gen_lab = 30 # number of sequences to generate per label
if purpose == 'artifact':
    for label in range(config.label_dim):
        print(f"doing label {label}")
        filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "evaluate/generate/" + config.run_name + '_lab' + str(label))

        if exists(filepath_for_artifacts) is False:
            os.mkdir(filepath_for_artifacts)

        for i in range(30):
            generate_f.generate_and_save(
                model=model, 
                epoch=latest_epoch + i + 1, 
                y_given=label, 
                config=config,
                log_to_wandb=False,
                single_epoch=filepath_for_artifacts,
                comic=True,
            )

if purpose == 'blind':
    set_of_blind_sequences = []
    for label in range(config.label_dim):
        one_label_seq = []
        for i in range(num_gen_lab):
            x_create, y_title = generate_f.generate(
                                            model=model, 
                                            y_given=label, 
                                            )
            x_create_formatted = x_create[0].reshape((config.seq_len, -1, 3))
            x_create_formatted = x_create_formatted.cpu().data.numpy()
            one_label_seq.append(x_create_formatted) # shape [100, 40, 53, 3]
        
        set_of_blind_sequences.append(one_label_seq)

    set_of_blind_sequences = np.array(set_of_blind_sequences).reshape(-1, config.seq_len, 53, 3 )

    # make array of original labels with size [300, 1]
    associated_labels = np.concatenate((np.zeros(num_gen_lab), np.ones(num_gen_lab), 2 * np.ones(num_gen_lab)))
    
    # reshuffle sequences and labels along the first axis
    shuffler = np.random.permutation(len(associated_labels))
    associated_labels_shuffled = np.array(associated_labels[shuffler])
    set_of_blind_sequences_shuffled = np.array(set_of_blind_sequences[shuffler])

    # DO NOT RE-SAVE NEW SEQUENCES, MUST MATCH SEQUENCES IN APP
    # save shuffles sequences and labels. We will plot these sequences in the shuffled order
    np.save('shuffled_seq_789', set_of_blind_sequences_shuffled)
    np.save('shuffled_labels_789', associated_labels_shuffled)    

if purpose == 'one_move':
    
    filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "evaluate/one_move/" + config.load_from_checkpoint)
    
    if exists(filepath_for_artifacts) is False:
        os.mkdir(filepath_for_artifacts)
    
    generate_f.generate_and_save_one_move(
        model=model, 
        config=config,
        path=filepath_for_artifacts,
    )