"""Training functions."""

import logging

import artifact
import numpy as np
import torch
import wandb

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


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
    """Run training and track it with wandb.

    The loss is given average on the number of examples (i.e. sequences)
    that have been seen until the current minibatch.

    loss_epoch = average loss per sequence.
    """
    batch_ct = 0
    batch_valid_ct = 0
    for epoch in range(epochs):
        seqs_in_epoch_ct = 0  # number of examples (sequences) seen
        batchs_in_epoch_ct = 0

        # Training
        model = model.train()

        loss_epoch_total = 0
        for x in data_train_torch:
            batchs_in_epoch_ct += 1
            batch_ct += 1

            n_seqs_in_batch = len(x)
            x = x.to(DEVICE)

            loss = train_batch(x, model, optimizer, get_loss)
            loss_epoch_total += loss * n_seqs_in_batch

            seqs_in_epoch_ct += n_seqs_in_batch
            loss_epoch_per_seq = loss_epoch_total / seqs_in_epoch_ct
            # Report metrics every 25th batch
            if (batchs_in_epoch_ct + 1) % 25 == 0:
                train_log(loss_epoch_per_seq, batchs_in_epoch_ct, batch_ct, epoch)

        # Validation
        model = model.eval()
        seqs_valid_in_epoch_ct = 0  # number of examples (sequences) seen
        batchs_valid_in_epoch_ct = 0

        loss_valid_epoch_total = 0
        for x in data_valid_torch:
            batch_valid_ct += 1
            batchs_valid_in_epoch_ct += 1
            x = x.to(DEVICE)
            n_seqs_in_batch = len(x)

            loss_valid = valid_batch(x, model, get_loss)
            loss_valid_epoch_total += loss_valid * n_seqs_in_batch

            seqs_valid_in_epoch_ct += n_seqs_in_batch
            loss_valid_epoch_per_seq = loss_valid_epoch_total / seqs_valid_in_epoch_ct

            # Report metrics every 25th batch
            if (batchs_valid_in_epoch_ct + 1) % 25 == 0:
                valid_log(loss_valid_epoch_per_seq, batchs_valid_in_epoch_ct, epoch)

        # Testing
        index_of_chosen_seq = np.random.randint(0, data_test_torch.dataset.shape[0])
        logging.info(f"Test: Make stick video for seq of index {index_of_chosen_seq}")

        # Batch size is 1 for data_test_torch
        for i, x in enumerate(data_test_torch):
            if i == index_of_chosen_seq:
                x = x.to(DEVICE)
                x_recon, _, _, _ = model(x.float())
                break

        x_formatted = x.reshape((128, -1, 3))
        x_recon_formatted = x_recon.reshape((128, -1, 3))

        logging.info(f"Call animation function for epoch {epoch}")
        fname = artifact.animate_stick(
            x_formatted,
            epoch=epoch,
            index=index_of_chosen_seq,
            ghost=x_recon_formatted,
            dot_alpha=0.7,
            ghost_shift=0.2,
            figsize=(12, 8),
        )
        animation_artifact = wandb.Artifact("animation", type="video")
        animation_artifact.add_file(fname)
        wandb.log_artifact(animation_artifact)
        logging.info("Logged artifact to wandb")

    logging.info("Done training.")


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
    optimizer.zero_grad()

    x_recon, z, z_mean, z_logvar = model(x.float())
    assert x.shape == x_recon.shape

    loss = get_loss(model, x, x_recon, z, z_mean, z_logvar)
    assert len(loss.shape) == 0

    loss.backward()
    optimizer.step()
    return loss


def train_log(loss, batchs_in_epoch_ct, batch_ct, epoch):
    """Log epoch and train loss into wandb."""
    wandb.log({"epoch": epoch, "loss": loss}, step=batch_ct)
    batchs_str = str(batchs_in_epoch_ct).zfill(5)
    logging.info(f"Train (Epoch {epoch}): Loss/seq after {batchs_str} batchs: {loss}")


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


def valid_log(loss, batchs_in_epoch_ct, epoch):
    """Log validation loss to wandb."""
    wandb.log({"valid_loss": loss})
    batchs_str = str(batchs_in_epoch_ct).zfill(5)
    logging.info(
        f"# Valid (Epoch {epoch}): Loss/seq after {batchs_str} batches: {loss}"
    )
