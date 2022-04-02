"""Training functions."""

import artifact
import default_config
import numpy as np
import torch
import wandb
from torch.autograd import Variable


def get_loss(model, x, x_recon, z, z_mu, z_logvar):
    loss = torch.mean(model.elbo(x, x_recon, z, (z_mu, z_logvar)))
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
            x = Variable(x)
            x = x.to(default_config.device)

            loss = train_batch(x, model, optimizer, get_loss)
            loss_epoch += loss

            example_ct += len(x)  # add amount of examples in 1 batch
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        loss_epoch /= batch_ct  # get average loss/epoch

        # Run Validation
        model = model.eval()

        loss_valid_epoch = 0
        for x in data_valid_torch:
            x = Variable(x)
            x = x.to(default_config.device)

            loss_valid = valid_batch(
                x, model, get_loss
            )  # same as before, except no back propogation
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
        for x in data_test_torch:
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

    # Forward pass
    x_recon, z, z_mu, z_logvar = model(x.float())
    # x_recon_batch_first=x_recon.reshape(
    # (x_recon.shape[1], x_recon.shape[0],x_recon.shape[2]))
    loss = get_loss(model, x, x_recon, z, z_mu, z_logvar)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Optimizer takes step
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    """Log epoch and train loss into wandb."""
    # wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print("Loss after {} examples: {}".format(str(example_ct).zfill(5), loss))


def valid_batch(x, model, get_loss):

    # Forward pass
    x_recon, z, z_mu, z_logvar = model(x.float())
    # x_recon_batch_first=x_recon.reshape(
    # (x_recon.shape[1], x_recon.shape[0],x_recon.shape[2]))
    valid_loss = get_loss(model, x, x_recon, z, z_mu, z_logvar)

    return valid_loss


def valid_log(valid_loss, example_ct, epoch):
    # Where the magic happens
    # wandb.log({"valid_loss": valid_loss})
