"""Default configuration parameters.

PirouNet_watch
checkpoint_smaller_lstm_contd_epoch144
PirouNet_dance
checkpoint_15_perc_labelled_epoch489
"""
import torch

run_name = "train_PirouNet_dance"
load_from_checkpoint = "checkpoint_15_perc_labelled_epoch489"

# Wandb
project = "move_labelled_nina"
entity = "bioshape-lab"

# Hardware
which_device = "0"  # CHANGE BACK TO 1 FOR TRAINING (0 for metrics)
device = (
    torch.device("cuda:" + str(which_device))
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Training
epochs = 500
learning_rate = 3e-4
batch_size = 80
with_clip = False

# Input data
seq_len = 40
input_dim = 159
label_dim = 3
amount_of_labels = 1
effort = "time"
fraction_label = 0.789

# LSTM VAE architecture
kl_weight = 1
neg_slope = 0
n_layers = 5
h_dim = 100
latent_dim = 256

# Classifier architecture
h_dim_classif = 100
neg_slope_classif = 0
n_layers_classif = 2
