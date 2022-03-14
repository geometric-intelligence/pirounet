import logging
import torch

#Set the configuration of the model 
logging.info('Confirgure the run')
batchsize = 8
learning_rate= 3e-5
epochs = 10
seq_len=128

logging.info('Setup device')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')