import datasets
import default_config
import generate
import torch
import logging
import nn

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")



labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, \
    labels_valid, labelled_data_test, labels_test, unlabelled_data_test = \
    datasets.get_dgm_data(default_config)

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

# Save artifact
logging.info(f"Artifacts: Make stick videos for epoch {epoch}")
artifact_maker = generate.Artifact(model, epoch=epoch)
artifact_maker.recongeneral(labelled_data_valid, labels_valid)
artifact_maker.recongeneral(labelled_data_test, labels_test)
for label in range(1, default_config.label_features + 1):
    artifact_maker.generatecond(y_given=label)