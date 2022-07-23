"""
Configuration parameters for evaluation of a model.
Works in conjunction with the web labeling app (for
blind labeling and comparison of labels.)
"""
import default_config
import torch

##########################################################
# CUDA device for evaluation

which_device = 0
evaluation_device = (
    torch.device("cuda:" + str(which_device))
    if torch.cuda.is_available()
    else torch.device("cpu")
)
##########################################################
# Tiling hyperparameters

step_size = [0.1, 0.2, 0.2]
# heigh/width of tiles for labels 0, 1, 2

dances_per_tile = [3, 3, 1]
# minimum dances required to be in a high density
# neighborhood for labels 0, 1, 2

density_thresh = [0.8, 0.75, 0.75]
# minimum percentage of dances required to share
# the label for labels 0, 1, 2

##########################################################
# Data dimensions

label_dim = default_config.label_dim
seq_len = default_config.seq_len

##########################################################
# Run that is evaluated

load_from_checkpoint = default_config.load_from_checkpoint

##########################################################
# Evaluation hyperparameters

stat_sampling_size = 100
# how many iterations to bootstrap quantitative metrics

num_gen_cond_lab = 152
# how many sequences to conditionally generate per label for
# evaluation by quantiative metrics

num_random_artifacts = 30
# how many artifacts (animations) to save during random
# generation

num_cond_artifacts_per_lab = 10
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
