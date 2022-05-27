"""Training functions."""
import itertools
import logging
import os
import time

import default_config
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

import evaluate.generate_f as generate_f
import models.dgm_lstm_vae as dgm_lstm_vae
import models.utils as utils
import torch
import torch.autograd
import torch.nn
import wandb
from torch.autograd import Variable

CUDA_VISIBLE_DEVICES = 0, 1


def binary_cross_entropy(r, x):
    in_sum = x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8)

    return -torch.sum(in_sum, dim=-1)


def run_train_dgm(
    model,
    labelled_data_train,
    labels_train,
    unlabelled_data_train,
    labelled_data_valid,
    labels_valid,
    labelled_data_test,
    labels_test,
    unlabelled_data_test,
    optimizer,
    config,
):
    """Run training and track it with wandb.

    The loss is given average on the number of examples (i.e. sequences)
    that have been seen until the current minibatch.

    loss_epoch = average loss per sequence.
    """

    elbo = dgm_lstm_vae.SVI(model)

    alpha = 0.1 * len(unlabelled_data_train) / len(labelled_data_train)

    onehot_encoder = utils.make_onehot_encoder(config.label_dim)

    if config.load_from_checkpoint is not None:
        old_checkpoint_filepath = os.path.join(
            os.path.abspath(os.getcwd()), config.load_from_checkpoint
        )
        checkpoint = torch.load(old_checkpoint_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        latest_epoch = checkpoint["epoch"]

    else:
        latest_epoch = 0

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(config.epochs - latest_epoch):

        # Train
        model.train()
        total_loss, accuracy, recon_loss = (0, 0, 0)
        labloss, unlabloss, class_loss = (0, 0, 0)

        batches_seen = 0
        n_batches = len(unlabelled_data_train)

        for i_batch, (x, y, u) in enumerate(
            zip(
                itertools.cycle(labelled_data_train),
                itertools.cycle(labels_train),
                unlabelled_data_train,
            )
        ):

            # Wrap in variables
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.to(config.device), y.to(config.device)
            u = u.to(config.device)

            batches_seen += 1

            batch_one_hot = torch.zeros((1, 1, config.label_dim))
            for y_i in y:
                y_i_enc = onehot_encoder(y_i.item())
                y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
                batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)

            batch_one_hot = batch_one_hot[1:, :, :]
            y = batch_one_hot.to(config.device)

            if i_batch == 0:
                logging.info(f"Train minibatch x of shape: {x.shape}")

            L = -elbo(x, y)  # check that averaged on minibatch
            labloss += L
            U = -elbo(u)
            unlabloss += U

            logits = model.classify(x)

            # classification loss is averaged on minibatch
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
            class_loss += classication_loss

            # J_alpha is averaged on minibatch
            J_alpha = L - alpha * classication_loss + U

            J_alpha.backward()

            # gradient clipping
            if config.with_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()

            if (total_loss / batches_seen) > 1e20:
                logging.info(f"Loss exploded, skipping batch {batches_seen}")
                continue

            y_like_logits = y.reshape(y.shape[0], y.shape[-1])
            accuracy += torch.mean(
                (
                    torch.max(logits, 1).indices == torch.max(y_like_logits, 1).indices
                ).float()
            )

            if i_batch % 50 == 0 and i_batch != 0:
                logging.info(
                    f"Batch {i_batch}/{n_batches} at loss {total_loss / (batches_seen)}, accuracy {accuracy / (batches_seen)}"
                )

            if i_batch % 50 == 0 and i_batch != 0:
                logging.info(
                    f"Batch {i_batch}/total at loss {total_loss / (batches_seen)}, accuracy {accuracy / (batches_seen)}"
                )
                logging.info(f"        Recon lab-loss {labloss / (batches_seen)}")
                logging.info(f"        Recon unlab-loss {unlabloss / (batches_seen)}")

        logging.info(f"Epoch: {epoch}")
        logging.info(
            "[Train]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".format(
                total_loss / batches_seen, accuracy / batches_seen
            )
        )

        wandb.log({"epoch": epoch, "loss": total_loss / batches_seen}, step=epoch)
        wandb.log(
            {"epoch": epoch, "labelled_recon_loss": labloss / batches_seen}, step=epoch
        )
        wandb.log(
            {"epoch": epoch, "unlabelled_recon_loss": unlabloss / batches_seen},
            step=epoch,
        )
        wandb.log(
            {"epoch": epoch, "classification_loss": class_loss / batches_seen},
            step=epoch,
        )
        wandb.log({"epoch": epoch, "accuracy": accuracy / batches_seen}, step=epoch)

        # Validation
        total_loss_valid, accuracy_valid, recon_loss_valid = (0, 0, 0)
        model.eval()

        batches_v_seen = 0
        total_v_batches = len(labelled_data_valid)

        for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):

            x, y = Variable(x), Variable(y)
            x, y = x.to(config.device), y.to(config.device)

            batches_v_seen += 1

            batch_one_hot = torch.zeros((1, 1, config.label_dim))
            for y_i in y:
                y_i_enc = onehot_encoder(y_i.item())
                y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
                batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)
            batch_one_hot = batch_one_hot[1:, :, :]
            y = batch_one_hot.to(config.device)

            L = -elbo(x, y)
            U = -elbo(x)

            logits_v = model.classify(x)
            classication_loss_v = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha_v = L - alpha * classication_loss_v + U

            total_loss_valid += J_alpha_v.item()

            _, pred_idx = torch.max(logits_v, 1)
            _, lab_idx = torch.max(y, 1)

            y_like_logits = y.reshape(y.shape[0], y.shape[-1])
            accuracy_valid += torch.mean(
                (
                    torch.max(logits_v, 1).indices
                    == torch.max(y_like_logits, 1).indices
                ).float()
            )

            if i_batch % 5 == 0 and i_batch != 0:
                logging.info(
                    f"Batch {i_batch}/{total_v_batches} at VALID loss \
                    {total_loss_valid / batches_v_seen}, accuracy {accuracy_valid / batches_v_seen}"
                )
                logging.info(f"Artifacts for epoch {epoch}")
                generate_f.reconstruct(
                    model=model,
                    epoch=epoch,
                    input_data=x,
                    input_label=y,
                    purpose="valid",
                    config=config,
                )

        logging.info(f"Epoch: {epoch}")
        logging.info(
            "[Validate]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".format(
                total_loss_valid / batches_v_seen, accuracy_valid / batches_v_seen
            )
        )

        wandb.log(
            {"epoch": epoch, "valid_loss": total_loss_valid / batches_v_seen},
            step=epoch,
        )
        wandb.log(
            {"epoch": epoch, "valid_accuracy": accuracy_valid / batches_v_seen},
            step=epoch,
        )

        for label in range(config.label_dim):
            generate_f.generate_and_save(
                model=model, epoch=epoch, y_given=label, config=config
            )

        logging.info("Save a checkpoint.")
        checkpoint_filepath = os.path.join(
            os.path.abspath(os.getcwd()),
            "saved/checkpoint_{}_epoch{}.pt".format(config.run_name, epoch),
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": (total_loss / batches_seen),
                "accuracy": (accuracy / batches_seen),
                "valid_loss": (total_loss_valid / batches_v_seen),
                "valid_accuracy": (accuracy_valid / batches_v_seen),
            },
            checkpoint_filepath,
        )
