import models.utils as utils
import torch
import default_config as config

y = [0,1,2]

def batch_one_hot(y)
    onehot_encoder = utils.make_onehot_encoder(config.label_dim)
    batch_one_hot = torch.zeros((1, 1, config.label_dim))
    for y_i in y:
        y_i_enc = onehot_encoder(y_i)
        y_i_enc = y_i_enc.reshape((1, 1, config.label_dim))
        batch_one_hot = torch.cat((batch_one_hot, y_i_enc), dim=0)

    batch_one_hot = batch_one_hot[1:, :, :]
    return batch_one_hot

    
print(batch_one_hot)