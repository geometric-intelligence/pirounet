import logging
import torch

#Set the configuration of the model 
logging.info('Confirgure the run')
batch_size = 8
learning_rate= 3e-4
epochs = 20
seq_len=128
negative_slope = 0 #LeakyRelu
kl_weight = 0

logging.info('Setup device')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')