"""Architectures of DGM LSTM VAE."""

import torch.nn

import models.losses as losses
import models.utils as utils
from models.classifiers import LinearClassifier
from models.lstm_vae import LstmDecoder, LstmEncoder


class DeepGenerativeModel(torch.nn.Module):
    """
    M2 code replication from the paper
    'Semi-Supervised Learning with Deep Generative Models'
    (Kingma 2014) in PyTorch.
    The "Generative semi-supervised model" is a probabilistic
    model that incorporates label information in both
    inference and generation.

    Parameters
    ----------
    n_layers :          int
                        Number of LSTM layers.
    input_dim :         int
                        Number of features per pose,
                        keypoints * 3 dimensions.
    h_dim :             int
                        Number of nodes in hidden layers.
    latent_dim :        int
                        Dimension of latent space.   
    output_dim :        int
                        Number of features per output
                        pose, keypoints * 3 dimensions.
    seq_len :           int
                        Number of poses in a sequence.  
    neg_slope :         int
                        Slope for LeakyRelu activation.
    label_dim :         int
                        Amount of categorical labels.
    batch_size :        int
                        Amount of examples (sequences)
                        in a  batch.
    h_dim_classif :     int
                        Amout of nodes in the 
                        classifier's hidden layers.
    neg_slope_classif : int
                        Slope for LeakReLU activation
                        in classifier.
    n_layers_classif :  int
                        Amount of hidden linear 
                        layers in classifier.
    encoder :           object of the class torch.nn.Module
                        Choice of encoder
                        {"LstmEncoder"}
    decoder :           clobject of the class torch.nn.Moduleass
                        Choice of decoder
                        {"LstmDecoder"}
    """
    def __init__(
        self,
        n_layers,
        input_dim,
        h_dim,
        latent_dim,
        output_dim,
        seq_len,
        neg_slope,
        label_dim,
        batch_size,
        h_dim_classif,
        neg_slope_classif,
        n_layers_classif,
        encoder = None,
        decoder = None
    ):

        super(DeepGenerativeModel, self).__init__()
        self.label_dim = label_dim
        self.seq_len = seq_len

        if encoder is None :
            self.encoder = LstmEncoder(
                n_layers=n_layers,
                input_dim=input_dim,
                h_dim=h_dim,
                latent_dim=latent_dim,
                label_dim=label_dim,
            )

        if decoder is None :
            self.decoder = LstmDecoder(
                n_layers=n_layers,
                output_dim=output_dim,
                h_dim=h_dim,
                latent_dim=latent_dim,
                seq_len=seq_len,
                neg_slope=neg_slope,
                label_dim=label_dim,
                batch_size=batch_size,
            )

        self.classifier = LinearClassifier(
            input_dim=input_dim,
            h_dim=h_dim_classif,
            label_dim=label_dim,
            seq_len=seq_len,
            neg_slope=neg_slope_classif,
            n_layers=n_layers_classif,
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        """ Performs forward pass of the DGM.

        Parameters
        ----------
        x :     array
                Shape = [batchsize, seq_len, input_dim]
                Input batch of sequences.
        y :     array
                Shape = [batchsize, 1,label_dim]
                Input batch of labels.
        
        Returns
        ----------
        x_mu :  array
                Shape = [batchsize, seq_len, input_dim]
                Batch of reconstructed sequences.       
            
        """

        y_for_encoder = y.repeat((1, self.seq_len, 1))
        z, _, _ = self.encoder(
            torch.cat([x, y_for_encoder], dim=2).float()
            )

        y_for_decoder = y.reshape((y.shape[0], y.shape[-1]))
        x_mu = self.decoder(
            torch.cat([z, y_for_decoder], dim=1).float()
            )

        return x_mu

    def encode(self, x, y):
        """ Encode a sequence into the latent space.

        Parameters
        ----------
        x :         array
                    Shape = [batchsize, seq_len, input_dim]
                    Input bach of sequences.
        y :         array
                    Shape = [batchsize, 1,label_dim]
                    Input batch of corresponding labels.

        Returns
        ----------
        z :         array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Latent variable sampled from Gaussian 
                    distribution.

        z_mu :      array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Mean of the Gaussian distribution.
        z_log_var : array
                    Shape = [batch_size, latent_dim] (labeled)
                    OR
                    Shape = [batch_size * label_dim, latent_dim]
                    (unlabeled)
                    Log variance of the Gaussian distribution.
        """

        y_for_encoder = y.repeat((1, self.seq_len, 1))
        z, z_mu, z_log_var = self.encoder(
            torch.cat([x, y_for_encoder], dim=2).float()
            )

        return z, z_mu, z_log_var

    def classify(self, x):
        """Classify input x into logits.

        Parameters
        ----------
        x :         array
                    Shape = [batch_size, seq_len, input_dim]
                    Input batch of sequences.

        Returns
        -------
        logits :    array-like
                    Shape = [batch_size, label_dim]
                    Batch of normalized probability vectors.
        """
        logits, _ = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Decodes sampled variable from the latent space 
        to generate an x.

        Parameters
        ----------
        z :     array
                Shape = [batch_size, latent_dim]
                Latent normal variable (batch of).
        y :     array
                Shape = [batch_size, label_dim]
                One hot encoded label (batch of).

        Returns
        -------
        x :     array-like
                Shape = [batch_size, seq_len, input_dim]
                Output batch of sequences.
        """
        y = y.float()

        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class SVI(torch.nn.Module):
    """
    Stochastic variational inference (SVI)
    optimizer, responsible for computing the
    evidence lower bound during training.
    The original version in https://github.com
    /wohlert/semi-supervised-pytorch includes an 
    Importance Weighted Sampler to increase Monte 
    Carlo sampling dimension of x and y, but does 
    not use it. We exclude it here for clarity 
    purposes.

    Parameters
    ----------
    model : serialized object
            Semi-supervised model to be evaluated.
    """

    def __init__(self, model):
        super(SVI, self).__init__()
        self.model = model

    def forward(self, x, y=None, likeli_func=losses.reconstruction_loss):
        """
        Performs forward pass of the SVI calculation.

        Parameters
        ----------
        x :             array
                        Shape = [batch_size, seq_len, input_dim]
                        Input batch of sequences. 
        y :             array
                        Shape = [batch_size, label_dim]
                        Batch of one hots associated to sequences.
                        In the unlabeled case, None.
        likeli_func :   function
                        Likelihood given by p(x|y,z).

        Returns in labeled case
        -------
        L_elbo :        float
                        Mean ELBO for batch, equivalent to 
                        L(x,y) in the paper (see appendices
                        for complete derivation).


        Returns in unlabeled case
        -------
        U_elbo :        float
                        Mean ELBO for batch, equivalent to 
                        U(x,y) in the paper (see appendices
                        for complete derivation).
        """

        batch_size = x.shape[0]
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = utils.enumerate_discrete(xs, self.model.label_dim)
            ys = ys.reshape((ys.shape[0], 1, ys.shape[-1]))
            xs = xs.repeat(self.model.label_dim, 1, 1)

            assert xs.shape == (
                batch_size * self.model.label_dim,
                x.shape[1],
                x.shape[2],
            )

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -likeli_func(xs, reconstruction)

        # p(y)
        prior = -torch.squeeze(utils.log_standard_categorical(ys))

        # Equivalent to -L(x, y) for each batch
        L_elbo_per_batch = likelihood + prior - self.model.kl_weight * self.model.kl_divergence

        if is_labelled:
            assert L_elbo_per_batch.shape == (batch_size,)
            L_elbo = torch.mean(L_elbo_per_batch)
            return L_elbo

        logits = self.model.classify(x)

        L_elbo_per_batch = L_elbo_per_batch.reshape((batch_size, self.model.label_dim))

        # Calculate entropy H(q(y|x)) and sum over all labels
        # H(p) = - integral_x p(x) logp(x) dx
        H_per_batch = torch.log(logits + 1e-8)
        assert L_elbo_per_batch.shape == logits.shape
        assert H_per_batch.shape == logits.shape

        H_weighted = torch.sum(torch.mul(logits, H_per_batch), dim=-1)
        L_weighted = torch.sum(torch.mul(logits, L_elbo_per_batch), dim=-1)

        assert H_weighted.shape == (batch_size,)
        assert L_weighted.shape == (batch_size,)

        # Equivalent to -U(x)
        U_elbo_per_batch = L_elbo_per_batch - H_weighted
        U_elbo = torch.mean(U_elbo_per_batch)
        return U_elbo
