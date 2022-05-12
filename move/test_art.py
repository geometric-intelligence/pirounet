import logging
import os
import time

import datasets
import default_config
import generate_f
import nn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

import wandb

CUDA_VISIBLE_DEVICES = 0, 1

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

wandb.init(
    project="move_labelled",
    entity="bioshape-lab",
    config={
        "learning_rate": default_config.learning_rate,
        "epochs": default_config.epochs,
        "batch_size": default_config.batch_size,
        "seq_len": default_config.seq_len,
        "kl_weight": default_config.kl_weight,
        "negative_slope": default_config.negative_slope,
        "n_layers": default_config.n_layers,
        "h_features_loop": default_config.h_features_loop,
        "latent_dim": default_config.latent_dim,
        "label_features": default_config.label_features,
    },
)

(
    labelled_data_train,
    labels_train,
    unlabelled_data_train,
    labelled_data_valid,
    labels_valid,
    labelled_data_test,
    labels_test,
    unlabelled_data_test,
) = datasets.get_dgm_data(default_config)

logging.info("Initialize model")
model = nn.DeepGenerativeModel(
    n_layers=default_config.n_layers,
    input_features=3 * 53,
    h_features_loop=default_config.h_features_loop,
    latent_dim=default_config.latent_dim,
    output_features=3 * 53,
    seq_len=default_config.seq_len,
    negative_slope=default_config.negative_slope,
    label_features=default_config.label_features,
).to(DEVICE)
epoch = 0

# filepath = os.path.join(os.path.abspath(os.getcwd()), "saved/latest_checkpoint.pt")
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
# }, filepath)

# Save artifact
logging.info(f"Artifacts: Make stick videos for epoch {epoch}")
generate_f.recongeneral(model, epoch, labelled_data_valid, labels_valid, "valid")
generate_f.recongeneral(model, epoch, labelled_data_test, labels_test, "test")
for label in range(1, default_config.label_features + 1):
    generate_f.generatecond(model, epoch=epoch, y_given=label)
