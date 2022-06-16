"""Default configuration parameters.

From Pettee 2019, Beyond Imitation.
The final architecture for the sequence VAE also
comprises an encoder and a decoder, each with
- 3 LSTM layers with 384 nodes and
- 1 dense layer with 256 nodes and
- a ReLU activation function,
where 256 represents the dimensionality of the latent space.

The model was compiled with the Adam optimizer.

The VAE maps:

- inputs of shape (53 x 3 x l), where l is the fixed length of the movement sequence,
- to the (256 x l)-dimensional latent space
- and then back to their original dimensionality.

We used input sequences of length l = 128, which corresponds
to about 4 seconds of continuous movement.
"""
import torch

run_name = 'perc_labelled_sweep' # "hdim100_hclass100_batch40_lr3e4"
load_from_checkpoint = (
   None #"checkpoint_10_perc_labelled_epoch468"
)

# Hardware
which_device = "1"
device = (
    torch.device("cuda:"+str(which_device)) if torch.cuda.is_available() else torch.device("cpu")
)

# Training
epochs = 500
learning_rate = 3e-4  # 6e-6
batch_size = 80 #40
with_clip = False

# Input data
seq_len = 40
input_dim = 159
label_dim = 3
amount_of_labels = 1
effort = 'time'
fraction_label = 0.2

# LSTM VAE
kl_weight = 1
neg_slope = 0  # 0.1,0.5 LeakyRelu
n_layers = 5  # ,5,6
h_dim = 100
latent_dim = 256

# Classifier
classifier = 'linear'
h_dim_classif = 100
neg_slope_classif = 0  # 0.5 #0.1 # 0.05
n_layers_classif = 2
