"""
Configuration parameters for evaluation of a model.
Works in conjunction with the web labeling app (for
blind labeling and comparison of labels.)
"""
import default_config
import torch

eval_name = "PirouNet_dance"
# name of evaluation

##########################################################
# CUDA device for evaluation

which_device = 0
device = (
    torch.device("cuda:" + str(which_device))
    if torch.cuda.is_available()
    else torch.device("cpu")
)
##########################################################
# Tiling hyperparameters

step_size = [0.1, 0.2, 0.2]
# height/width of tiles for labels 0, 1, 2

dances_per_tile = [3, 3, 1]
# minimum dances required to be in a high density
# neighborhood for labels 0, 1, 2

density_thresh = [0.8, 0.75, 0.75]
# minimum percentage of dances required to share
# the label for labels 0, 1, 2

##########################################################
# Data dimensions

input_dim = default_config.input_dim
label_dim = default_config.label_dim
seq_len = default_config.seq_len
latent_dim = default_config.latent_dim

##########################################################
# Run that is evaluated

load_from_checkpoint = default_config.load_from_checkpoint

##########################################################
# Labels to be compared

human_labels = "data/labels_shuffled_neighb.csv"
pirounet_labels = "data/shuffled_labels_variations.npy"

##########################################################
# Evaluation hyperparameters

stat_sampling_size = 1
# how many iterations to bootstrap quantitative metrics

num_gen_cond_lab = 152
# how many sequences to conditionally generate per label for
# evaluation by quantiative metrics

num_random_artifacts = 2
# how many artifacts (animations) to save during random
# generation

num_cond_artifacts_per_lab = 1
# how many artifacts (animations) to save per label during
# conditional generation

##########################################################
# Desired metrics

quanti_gen_recon_metrics = True
quali_generation_metrics = True
quali_recon_metrics = True
test_entanglement = True
generate_for_blind_labeling = True
plot_recognition_accuracy = True
plot_classification_accuracy = True
plot_self_confusion = True
plot_latent_space = True
