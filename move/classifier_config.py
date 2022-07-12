"""Default configuration parameters for training seperate classifier/

The Classifier maps:

- inputs of shape (53 x 3 x l), where l is the fixed length of the movement sequence,
- to logits with shape [1, label_dim]


"""
import torch

run_name = 'classifier_metrics_test' # "hdim100_hclass100_batch40_lr3e4"
load_from_checkpoint = (
   "checkpoint_classifier_fid_epoch206"
)

# Hardware
which_device = "0"
device = (
    torch.device("cuda:"+str(which_device)) if torch.cuda.is_available() else torch.device("cpu")
)

# Training
epochs = 400
learning_rate = 3e-4  # 6e-6
batch_size = 55

# Input data
seq_len = 40
input_dim = 159
label_dim = 3
amount_of_labels = 1
effort = 'time'
fraction_label = 0.92

# Classifier's linear layers
h_dim_class = 250
neg_slope_classif = 0.42950429805116874  # 0.5 #0.1 # 0.05
n_layers_class = 13

# Classifier's LSTM layers
h_dim = 100
n_layers = 5