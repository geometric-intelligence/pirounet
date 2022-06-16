"""Default configuration parameters for training seperate classifier/

The Classifier maps:

- inputs of shape (53 x 3 x l), where l is the fixed length of the movement sequence,
- to logits with shape [1, label_dim]


"""
import torch

run_name = 'classifier_sweep_no_gen' # "hdim100_hclass100_batch40_lr3e4"
load_from_checkpoint = (
   None #"checkpoint_smaller_lstm_contd_epoch144"
)

# Hardware
which_device = "0"
device = (
    torch.device("cuda:"+str(which_device)) if torch.cuda.is_available() else torch.device("cpu")
)

# Training
epochs = 150
learning_rate = 3e-4  # 6e-6
batch_size = 40

# Input data
seq_len = 40
input_dim = 159
label_dim = 3
amount_of_labels = 1
effort = 'time'

# Classifier
h_dim_classif = 100
neg_slope_classif = 0  # 0.5 #0.1 # 0.05
n_layers_classif = 2
