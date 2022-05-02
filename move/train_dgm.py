"""Training functions."""

import logging
import os
import time

import artifact
import numpy as np
import torch
import wandb
from torch.autograd import Variable
from nn import SVI, ImportanceWeightedSampler



DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

def run_train_dgm(
    model,
    labelled_data_train,
    unlabelled_data_train,
    labelled_data_valid,
    labelled_data_test,
    unlabelled_data_test,
    optimizer,
    epochs,
):
    """Run training and track it with wandb.

    The loss is given average on the number of examples (i.e. sequences)
    that have been seen until the current minibatch.

    loss_epoch = average loss per sequence.
    """

    elbo = SVI(model, likelihood=binary_cross_entropy, sampler=sampler)
    sampler = ImportanceWeightedSampler(mc=1, iw=1)
    alpha = 0.1 * len(unlabelled_data_train) / len(labelled_data_train)

    batch_ct = 0
    seq_ct = 0
    batch_valid_ct = 0
    seq_valid_ct = 0

    for epoch in range(epochs):

        #Train
        model.train()
        total_loss, accuracy = (0, 0)
        seq_ct_in_epoch = 0
        for i_batch, (x, y), (u, _) in enumerate(zip(cycle(labelled_data_train), unlabelled_data_train)):
            # Wrap in variables
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.to(DEVICE), y.to(DEVICE)
            u = u.to(DEVICE)

            if i_batch == 0:
                logging.info(f"Train minibatch x of shape: {x.shape}")
            batch_ct += 1
            seq_ct += len(x)
            seq_ct_in_epoch += len(x)

            L = -elbo(x, y) #loss associated to data x with label y
            U = -elbo(u) #loss associated to unlabelled data (why does SVI work here?)

            # Add auxiliary classification loss q(y|x)
            #pushes useful gradients to lower layers in order to reduce vanishing gradients
            logits = model.classify(x)
            
            # Regular cross entropy
            #comes from given probability that the data x has label y
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L - alpha * classication_loss + U #loss for this "batch" (?)
            
                 #accounts for if auxiliary classification is good 

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.data[0]
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())
                
            if epoch % 1 == 0:
                m = len(unlabelled)
                print("Epoch: {}".format(epoch))
                print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
                wandb.log({"epoch": epoch, "loss": total_loss}, step=epoch)
                wandb.log({"epoch": epoch, "accuracy": accuracy}, step=epoch)

        #Validation
        total_loss_valid, accuracy_valid = (0, 0)
        model.eval()
        for i_batch, (x, y) in enumerate(zip(cycle(labelled_data_valid))):

            x, y = Variable(x), Variable(y)
            x, y = x.to(DEVICE), y.to(DEVICE)

            L = -elbo(x, y)
            U = -elbo(x)

            logits_v = model.classify(x) 
            classication_loss_v = torch.sum(y * torch.log(logits_v + 1e-8), dim=1).mean()

            J_alpha_v = L - alpha * classication_loss_v + U  

            total_loss_valid += J_alpha_v.data[0]

            _, pred_idx = torch.max(logits_v, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy_valid += torch.mean((torch.max(logits_v, 1)[1].data == torch.max(y, 1)[1].data).float())

            m = len(validation)
            print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss_valid / m, accuracy_valid / m))
            wandb.log({"epoch": epoch, "valid_loss": total_loss_valid}, step=epoch)
            wandb.log({"epoch": epoch, "valid_accuracy": accuracy_valid}, step=epoch)
        

