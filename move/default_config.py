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
which_device = "0"
run_name = "3e-4"
label_features = 4
amount_of_labels = 1

batch_size = 80
learning_rate = 3e-4 #6e-6
epochs = 400 
seq_len = 40
negative_slope = 0  # 0.1,0.5 LeakyRelu
kl_weight = 0
n_layers = 5 #,5,6
h_dim = 384
latent_dim = 256
input_features = 159
h_dim_classif = 384
neg_slope_classif = 0 #0.5 #0.1 # 0.05
n_layers_classif = 2
