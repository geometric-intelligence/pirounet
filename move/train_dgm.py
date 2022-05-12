"""Training functions."""
import logging
import os
import time
from itertools import cycle

import default_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

import generate_f
import torch
import utils
from models.dgm import SVI
from torch.autograd import Variable

import wandb

CUDA_VISIBLE_DEVICES = 0, 1
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


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
    epochs,
    label_features,
    checkpoint=False,
):
    """Run training and track it with wandb.

    The loss is given average on the number of examples (i.e. sequences)
    that have been seen until the current minibatch.

    loss_epoch = average loss per sequence.
    """

    elbo = SVI(model)

    alpha = 0.1 * len(unlabelled_data_train) / len(labelled_data_train)

    onehot_encoder = utils.make_onehot_encoder(label_features)

    now = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_filepath = os.path.join(
        os.path.abspath(os.getcwd()),
        "saved/checkpoint_{}_{}.pt".format(now, default_config.run_name),
    )

    if checkpoint is True:
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        latest_epoch = checkpoint["epoch"]
        total_loss = checkpoint["loss"]

    else:
        latest_epoch = 0

    for epoch in range(epochs - latest_epoch):

        # Train
        model.train()
        total_loss, accuracy = (0, 0)

        batches_seen = 0

        for i_batch, (x, y, u) in enumerate(zip(cycle(labelled_data_train),
                cycle(labels_train), unlabelled_data_train)):
            # Wrap in variables
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.to(DEVICE), y.to(DEVICE)
            u = u.to(DEVICE)
            
            batches_seen += 1

            batch_one_hot = torch.zeros((1, 1, label_features))
            for y_i in y:
                y_i_enc = onehot_encoder(y_i.item())
                y_i_enc = y_i_enc.reshape((1, 1, label_features))
                batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)

            batch_one_hot = batch_one_hot[1:, :, :]
            y = batch_one_hot.to(DEVICE)

            if i_batch == 0:
                logging.info(f"Train minibatch x of shape: {x.shape}")

            L = -elbo(x, y)  # check that averaged on minibatch
            U = -elbo(u)

            logits = model.classify(x)

            # classification loss is averaged on minibatch
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            # J_alpha is averaged on minibatch
            J_alpha = L - alpha * classication_loss + U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()
            
            y_like_logits = y.reshape(y.shape[0], y.shape[-1])
            accuracy += torch.mean(
                (torch.max(logits, 1).indices
                    == torch.max(y_like_logits, 1).indices).float())

            if i_batch % 50 == 0 and i_batch != 0 :
                logging.info(f"Batch {i_batch}/total at loss {total_loss / (batches_seen)}, accuracy {accuracy / (batches_seen)}")

        logging.info(f"Epoch: {epoch}")
        logging.info("[Train]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".
                format(total_loss / batches_seen, accuracy / batches_seen))

        wandb.log({"epoch": epoch, "loss": total_loss / batches_seen}, step=epoch)
        wandb.log({"epoch": epoch, "accuracy": accuracy / batches_seen}, step=epoch)

        # Validation
        total_loss_valid, accuracy_valid = (0, 0)
        model.eval()

        batches_v_seen = 0
        total_v_batches = len(labelled_data_valid)
        
        for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):

            x, y = Variable(x), Variable(y)
            x, y = x.to(DEVICE), y.to(DEVICE)

            batches_v_seen += 1

            batch_one_hot = torch.zeros((1, 1, label_features))
            for y_i in y:
                y_i_enc = onehot_encoder(y_i.item())
                y_i_enc = y_i_enc.reshape((1, 1, label_features))
                batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)
            batch_one_hot = batch_one_hot[1:, :, :]
            y = batch_one_hot.to(DEVICE)

            L = -elbo(x, y)
            U = -elbo(x)

            logits_v = model.classify(x)
            classication_loss_v = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha_v = L - alpha * classication_loss_v + U

            total_loss_valid += J_alpha_v.item()

            _, pred_idx = torch.max(logits_v, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy_valid += torch.mean(
                (torch.max(logits_v.t(), 1).indices
                    == torch.max(y, 1).indices).float())

            if i_batch % 50 == 0 and i_batch != 0 :
                logging.info(f"Batch {i_batch}/total at VALID loss \
                    {total_loss_valid / batches_v_seen}, accuracy {accuracy_valid / batches_v_seen}")

        logging.info(f"Epoch: {epoch}")
        logging.info("[Validate]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".
                format(total_loss_valid / batches_v_seen, accuracy_valid / batches_v_seen))

        wandb.log({"epoch": epoch, "valid_loss": total_loss_valid / batches_v_seen}, step=epoch)
        wandb.log({"epoch": epoch, "valid_accuracy": accuracy_valid / batches_v_seen}, step=epoch)

        logging.info(f"Artifacts: Make stick videos for epoch {epoch}")
        generate_f.recongeneral(model, epoch, labelled_data_valid, labels_valid, 'valid')
        generate_f.recongeneral(model, epoch, labelled_data_test, labels_test, 'test')
        for label in range(1, default_config.label_features + 1):
            generate_f.generatecond(model, epoch=epoch, y_given=label)

        logging.info('Save a checkpoint.')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'accuracy': accuracy
        }, checkpoint_filepath)

