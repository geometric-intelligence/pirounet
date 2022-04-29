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


def get_loss(model, x, x_recon, z, z_mean, z_logvar):
    """Return loss as ELBO averaged on the minibatch.

    This gives a loss averaged on all sequences of the minibatch,
    i.e. a loss per sequence.
    """
    loss = torch.mean(model.elbo(x, x_recon, z, (z_mean, z_logvar)))
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

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

    model = model.cuda()
    #Minimizing KL divergence corresponds to maximizing ELBO and vice versa.
    elbo = SVI(model, likelihood=binary_cross_entropy, sampler=sampler)
    #SVI (Stochastic Variational Inference) is alternative to MCMC
    sampler = ImportanceWeightedSampler(mc=1, iw=1)


    for epoch in range(epochs):
        model.train()
        total_loss, accuracy = (0, 0)
        for i_batch, (x, y), (u, _) in enumerate(zip(cycle(labelled_data), unlabelled_data)):
            # Wrap in variables
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.device(), y.cuda(device=0)
            u = u.cuda(device=0)

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
            model.eval()
            m = len(unlabelled)
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
            
            #Validation
            total_loss, accuracy = (0, 0)
            for x, y in validation: #again, is this batch loop?
                x, y = Variable(x), Variable(y)
                x, y = x.cuda(device=0), y.cuda(device=0)

                L = -elbo(x, y)
                U = -elbo(x)

                logits = model.classify(x) #how is this different from auxiliary classification?
                classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                J_alpha = L + alpha * classication_loss + U  #why is classificaiton loss positive here?

                total_loss += J_alpha.data[0]

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

            m = len(validation)
            print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

        

