"""Training functions."""
import itertools
import logging
import os
from os.path import exists

import classifier_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = classifier_config.which_device

import evaluate.generate_f as generate_f
import models.dgm_lstm_vae as dgm_lstm_vae
import models.utils as utils

import torch
import torch.autograd
import torch.nn
from torch.autograd import Variable
import wandb


def run_train_classifier(
    model,
    labelled_data_train,
    labels_train,
    labelled_data_valid,
    labels_valid,
    optimizer,
    config,
):
    """Run training and track it with wandb.

    The loss is given average on the number of examples (i.e. sequences)
    that have been seen until the current minibatch.

    loss_epoch = average loss per sequence.
    """
    # if config.load_from_checkpoint is not None:
    #     old_checkpoint_filepath = os.path.join(
    #         os.path.abspath(os.getcwd()), "saved/classifier/" + config.load_from_checkpoint + ".pt"
    #     )
    #     checkpoint = torch.load(old_checkpoint_filepath)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     print('loaded from checkpoint')
    
    # if config.load_from_checkpoint is None:
    #     print('from scratch')

    for epoch in range(config.epochs):

        # Train
        model.train()
        total_loss, accuracy = 0,0
        class_loss = 0

        batches_seen = 0
        n_batches = len(labelled_data_train)
        n_batches_valid = len(labelled_data_valid)

        for i_batch, (x, y) in enumerate(zip(labelled_data_train, labels_train)):

            # Wrap in variables
            x, y = Variable(x), Variable(y)
            x, y = x.to(config.device), y.to(config.device)

            batches_seen += 1

            batch_one_hot = utils.batch_one_hot(y, config.label_dim)
            y = batch_one_hot.to(config.device)

            if i_batch == 0:
                logging.info(f"Train minibatch x of shape: {x.shape}")

            logits, _ = model.forward(x)

            # classification loss is averaged on minibatch
            classification_loss = - torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
            class_loss += classification_loss

            # J_alpha is averaged on minibatch
            J_alpha = classification_loss

            J_alpha.backward()
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

        logging.info(f"Epoch: {epoch}")
        logging.info(
            "[Train]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".format(
                total_loss / batches_seen, accuracy / batches_seen
            )
        )

        wandb.log({"epoch": epoch, "loss": total_loss / batches_seen}, step=epoch)
        wandb.log(
            {"epoch": epoch, "classification_loss": class_loss / batches_seen},
            step=epoch,
        )
        wandb.log({"epoch": epoch, "accuracy": accuracy / batches_seen}, step=epoch)

        # Validation
        total_loss_valid, accuracy_valid = 0,0
        model.eval()

        batches_v_seen = 0
        total_v_batches = len(labelled_data_valid)

        for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):

            x, y = Variable(x), Variable(y)
            x, y = x.to(config.device), y.to(config.device)

            batches_v_seen += 1

            batch_one_hot = utils.batch_one_hot(y, config.label_dim)
            y = batch_one_hot.to(config.device)

            logits_v, _ = model.forward(x)
            classication_loss_v = torch.sum(y * torch.log(logits_v + 1e-8), dim=1).mean()

            J_alpha_v = classication_loss_v

            total_loss_valid += J_alpha_v.item()

            y_like_logits = y.reshape(y.shape[0], y.shape[-1])
            accuracy_valid += torch.mean(
                (
                    torch.max(logits_v, 1).indices
                    == torch.max(y_like_logits, 1).indices
                ).float()
            )

            if n_batches_valid <= 5 and i_batch != 0:
                logging.info(
                    f"Valid batch {i_batch}/{total_v_batches} at loss \
                    {total_loss_valid / batches_v_seen}, accuracy {accuracy_valid / batches_v_seen}"
                )
            if n_batches > 5 and i_batch != 0:
                if i_batch % 5 == 0:
                    logging.info(
                        f"Valid batch {i_batch}/{total_v_batches} at loss \
                        {total_loss_valid / batches_v_seen}, accuracy {accuracy_valid / batches_v_seen}"
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

        logging.info("Save a checkpoint.")
        checkpoint_filepath = os.path.join(
            os.path.abspath(os.getcwd()),
            "saved/classifier/checkpoint_{}_epoch{}.pt".format(config.run_name, epoch),
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
