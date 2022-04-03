"""Training functions."""

import artifact
import default_config
import numpy as np
import torch
import wandb
from torch.autograd import Variable


def get_loss(model, x, x_recon, z, z_mean, z_logvar):
    """Return loss as ELBO averaged on the minibatch."""
    loss = torch.mean(model.elbo(x, x_recon, z, (z_mean, z_logvar)))
    return loss


def run_train(
    model,
    data_train_torch,
    data_valid_torch,
    data_test_torch,
    get_loss,
    optimizer,
    epochs,
):
    """Run training and track it with wandb."""
    example_ct = 0  # number of examples seen
    batch_ct = 0
    example_ct_valid = 0  # number of examples seen
    batch_ct_valid = 0
    for epoch in range(epochs):
        # Train
        model = model.train()

        loss_epoch = 0
        for x in data_train_torch:
            x = Variable(x)  # TODO: Do we need this?
            x = x.to(default_config.device)

            loss = train_batch(x, model, optimizer, get_loss)
            loss_epoch += loss * len(x)

            example_ct += len(x)  # add amount of examples in 1 batch
            loss_epoch /= (
                example_ct  # TODO: Sum of average of losses is not average of loss
            )

            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        loss_epoch /= batch_ct  # get average loss/epoch
        # TODO: Check: already summed?

        # Run Validation
        model = model.eval()

        loss_valid_epoch = 0
        for x in data_valid_torch:
            x = Variable(x)
            x = x.to(default_config.device)

            loss_valid = valid_batch(x, model, get_loss)
            loss_valid_epoch += loss_valid

            example_ct_valid += len(x)  # add amount of examples in 1 batch
            batch_ct_valid += 1

            # Report metrics every 25th batch
            if ((batch_ct_valid + 1) % 25) == 0:
                valid_log(loss_valid, example_ct_valid, epoch)

        # Run testing
        # Make and log artifact at the end of each epoch (stick-figure video)
        index_of_chosen_seq = np.random.randint(0, data_test_torch.dataset.shape[0])
        print("INDEX OF TESTING SEQUENCE IS {}".format(index_of_chosen_seq))
        i = 0
        for x in data_test_torch:  # minibatch: (1, 128, 53*3) batchsize is one
            i += 1

            if i == index_of_chosen_seq:
                print("Found test sequence. Running it through model")
                x = Variable(x)
                x = x.to(default_config.device)
                x_input = x
                x_recon, z, z_mu, z_logvar = model(x.float())
                print("Ran it through model")

            else:
                pass

        x_input_formatted = x_input.reshape((128, 53, 3))
        x_recon_formatted = x_recon.reshape((128, 53, 3))

        _ = artifact.animate_stick(
            x_input_formatted,
            epoch=epoch,
            index=index_of_chosen_seq,
            ghost=x_recon_formatted,
            dot_alpha=0.7,
            ghost_shift=0.2,
            figsize=(12, 8),
        )
        print("Called animation function for epoch {}".format(epoch + 1))

    print("done training")


def train_batch(x, model, optimizer, get_loss):
    """Perform a forward pass at training time.

    The loss is backpropagated at training time.

    Parameters
    ----------
    x : array-like, shape=[batch_size, seq_len, 3*n_joints]
        Input to the model.
    model : torch.nn.Module
        Model performing the forward pass.
    optimizer :
    get_loss : callable
        Function defining the loss.

    Returns
    -------
    loss : float-like, shape=[]
        Loss as computed through get_loss on
        an input minibatch.
    """
    x_recon, z, z_mean, z_logvar = model(x.float())
    # TODO: compare x and x recon here, especially look if x is not constant.
    print("\n\nabout x recon:")
    print(x_recon.shape)
    print(x_recon[0, :10, :6])
    print("about x:")
    print(x.shape)
    print(x[0, :10, :6])
    print("\n\n")
    loss = get_loss(model, x, x_recon, z, z_mean, z_logvar)
    print("about loss")
    print(loss.shape)
    # Backward pass
    # TODO: should the optimizer be put at 0 there?
    optimizer.zero_grad()
    loss.backward()

    # Optimizer takes step
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    """Log epoch and train loss into wandb."""
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print("Loss after {} examples: {}".format(str(example_ct).zfill(5), loss))


def valid_batch(x, model, get_loss):
    """Perform a forward pass at validation time.

    The loss is not backpropagated at validation time.

    Parameters
    ----------
    x : array-like, shape=[batch_size, seq_len, 3*n_joints]
        Input to the model.
    model : torch.nn.Module
        Model performing the forward pass.
    get_loss : callable
        Function defining the loss.

    Returns
    -------
    valid_loss : float-like
        Validation loss as computed through get_loss on
        an input minibatch.
    """
    x_recon, z, z_mean, z_logvar = model(x.float())
    valid_loss = get_loss(model, x, x_recon, z, z_mean, z_logvar)
    return valid_loss


def valid_log(valid_loss, example_ct, epoch):
    """Log validation loss to wandb."""
    wandb.log({"valid_loss": valid_loss})
