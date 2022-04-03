"""Training functions."""

import logging
import os
import time

import artifact
import numpy as np
import torch
import wandb

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def get_loss(model, x, x_recon, z, z_mean, z_logvar):
    """Return loss as ELBO averaged on the minibatch.

    This gives a loss averaged on all sequences of the minibatch,
    i.e. a loss per sequence.
    """
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
    seq_ct = 0
    batch_valid_ct = 0
    seq_valid_ct = 0
    for epoch in range(epochs):
        # Training
        model = model.train()
        seq_ct_in_epoch = 0  # number of examples (sequences) seen
        loss_epoch_total = 0
        for i_batch, x in enumerate(data_train_torch):
            if i_batch == 0:
                logging.info(f"Train minibatch x of shape: {x.shape}")
            batch_ct += 1
            seq_ct += len(x)
            seq_ct_in_epoch += len(x)

            x = x.to(DEVICE)
            loss = train_batch(x, model, optimizer, get_loss)
            loss_epoch_total += loss * len(x)

            if i_batch % 25 == 0:
                batchs_str = str(i_batch).zfill(5)
                loss_epoch_per_seq = loss_epoch_total / seq_ct_in_epoch
                logging.info(
                    f"Train (Epoch {epoch}): "
                    f"Loss/seq after {batchs_str} batchs: {loss_epoch_per_seq}"
                )
                train_log(loss_epoch_per_seq, seq_ct, epoch)

        # Validation
        model = model.eval()
        seq_valid_ct_in_epoch = 0  # number of examples (sequences) seen
        loss_valid_epoch_total = 0
        for i_batch, x in enumerate(data_valid_torch):
            if i_batch == 0:
                logging.info(f"Valid minibatch x of shape: {x.shape}")
            batch_valid_ct += 1
            seq_valid_ct += len(x)
            seq_valid_ct_in_epoch += len(x)

            x = x.to(DEVICE)
            loss_valid = valid_batch(x, model, get_loss)
            loss_valid_epoch_total += loss_valid * len(x)

            if i_batch % 25 == 0:
                batchs_str = str(i_batch).zfill(5)
                loss_valid_epoch_per_seq = (
                    loss_valid_epoch_total / seq_valid_ct_in_epoch
                )
                logging.info(
                    f"# Valid (Epoch {epoch}): "
                    f"Loss/seq after {batchs_str} batches: {loss_valid_epoch_per_seq}"
                )
                valid_log(loss_valid_epoch_per_seq, seq_valid_ct, epoch)

        # Artifacts
        logging.info(f"Artifacts: Make stick videos for epoch {epoch}")
        filepath = os.path.join(os.path.abspath(os.getcwd()), "animations")
        now = time.strftime("%Y%m%d_%H%M%S")

        name = f"train_artifact_epoch_{epoch}_on_{now}.gif"
        fname = os.path.join(filepath, name)
        train_artifact(model=model, data_train_torch=data_train_torch, fname=fname)

        seq_index = np.random.randint(0, data_test_torch.dataset.shape[0])
        name = f"test_artifact_epoch_{epoch}_index_{seq_index}_on_{now}.gif"
        fname = os.path.join(filepath, name)
        test_artifact(
            model=model,
            data_test_torch=data_test_torch,
            fname=fname,
            seq_index=seq_index,
        )

    logging.info("Done training.")


def train_artifact(model, data_train_torch, fname):
    """Make stick video on seq from train set."""
    for x in data_train_torch:
        x = x.to(DEVICE)
        x_recon, _, _, _ = model(x.float())
        break

    _, seq_len, _ = x.shape
    x_formatted = x[0].reshape((seq_len, -1, 3))
    x_recon_formatted = x_recon[0].reshape((seq_len, -1, 3))

    fname = artifact.animate_stick(
        x_recon_formatted,
        fname=fname,
        ghost=x_formatted,
        dot_alpha=0.7,
        ghost_shift=0.2,
    )
    animation_artifact = wandb.Artifact("animation", type="video")
    animation_artifact.add_file(fname)
    wandb.log_artifact(animation_artifact)
    logging.info("Logged train artifact to wandb.")


def test_artifact(model, data_test_torch, fname, seq_index):
    """Make stick video on seq from test set."""
    # Batch size is 1 for data_test_torch
    for i, x in enumerate(data_test_torch):
        if i == seq_index:
            x = x.to(DEVICE)
            x_recon, _, _, _ = model(x.float())
            break

    _, seq_len, _ = x.shape
    x_formatted = x.reshape((seq_len, -1, 3))
    x_recon_formatted = x_recon.reshape((seq_len, -1, 3))

    fname = artifact.animate_stick(
        x_recon_formatted,
        fname=fname,
        ghost=x_formatted,
        dot_alpha=0.7,
        ghost_shift=0.2,
    )
    animation_artifact = wandb.Artifact("animation", type="video")
    animation_artifact.add_file(fname)
    wandb.log_artifact(animation_artifact)
    logging.info("Logged test artifact to wandb.")


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


def train_log(loss, seq_ct, epoch):
    """Log epoch and train loss into wandb."""
    wandb.log({"epoch": epoch, "loss": loss}, step=seq_ct)


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


def valid_log(valid_loss, seq_ct, epoch):
    """Log validation loss to wandb."""
    wandb.log({"valid_loss": valid_loss})
