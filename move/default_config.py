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
which_device = "1"
run_name = "sep_layers_3e4NOCLIP_alpha001"
load_from_checkpoint = (
    None  # "saved/checkpoint_nan_enc_load_debug_prints_nonclipped_epoch19.pt"
)
amount_of_labels = 1

# Training
epochs = 400
learning_rate = 3e-4  # 6e-6
batch_size = 2  # 80
with_clip = False

# Input data
seq_len = 40
input_dim = 159
label_dim = 4
amount_of_labels = 1

# LSTM VAE
kl_weight = 0
neg_slope = 0  # 0.1,0.5 LeakyRelu
n_layers = 5  # ,5,6
h_dim = 8  # 384
latent_dim = 8  # 256

# Classifier
h_dim_classif = 8  # 384
neg_slope_classif = 0  # 0.5 #0.1 # 0.05
n_layers_classif = 1  # 2
