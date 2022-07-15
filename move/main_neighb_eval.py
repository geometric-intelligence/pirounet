import os

import classifier_config
import datasets
import default_config
import evaluate.generate_f as generate_f
import evaluate.metrics as metrics
import matplotlib.pyplot as plt
import models.classifiers as classifiers
import models.dgm_lstm_vae as dgm_lstm_vae
import models.utils as utils
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.autograd import Variable

# Run this script to get all evaluation metrics necessary
# for row in Table (except Danceability and Recognition Accuracy).
# Steps:
#   1. Change checkpoint in default_config and change any hyperparams to match
#   2. Set device to 0
#   3. Run main_evaluate.py

evaluation_device = (
    torch.device("cuda:" + str(classifier_config.which_device))
    if torch.cuda.is_available()
    else torch.device("cpu")
)

fid_classifier_model = classifiers.FID_LinearClassifier(
    input_dim=classifier_config.input_dim,
    h_dim=classifier_config.h_dim_class,
    label_dim=classifier_config.label_dim,
    seq_len=classifier_config.seq_len,
    neg_slope=classifier_config.neg_slope_classif,
    n_layers=classifier_config.n_layers_class,
).to(evaluation_device)

classif_old_checkpoint_filepath = os.path.join(
    os.path.abspath(os.getcwd()),
    "saved/classifier/" + classifier_config.load_from_checkpoint + ".pt",
)
classif_checkpoint = torch.load(classif_old_checkpoint_filepath)
fid_classifier_model.load_state_dict(classif_checkpoint["model_state_dict"])

model = dgm_lstm_vae.DeepGenerativeModel(
    n_layers=default_config.n_layers,
    input_dim=default_config.input_dim,
    h_dim=default_config.h_dim,
    latent_dim=default_config.latent_dim,
    output_dim=default_config.input_dim,
    seq_len=default_config.seq_len,
    neg_slope=default_config.neg_slope,
    label_dim=default_config.label_dim,
    batch_size=default_config.batch_size,
    h_dim_classif=default_config.h_dim_classif,
    neg_slope_classif=default_config.neg_slope_classif,
    n_layers_classif=default_config.n_layers_classif,
    classifier=default_config.classifier,
).to(evaluation_device)

old_checkpoint_filepath = os.path.join(
    os.path.abspath(os.getcwd()), "saved/" + default_config.load_from_checkpoint + ".pt"
)
checkpoint = torch.load(old_checkpoint_filepath)
model.load_state_dict(checkpoint["model_state_dict"])

(
    labelled_data_train,
    labels_train,
    unlabelled_data_train,
    labelled_data_valid,
    labels_valid,
    labelled_data_test,
    labels_test,
    unlabelled_data_test,
) = datasets.get_model_data(default_config)

####################################################

# BOOTSTRAP LOOP TO TAKE DISTRIBUTION ON STATS

stat_sampling_size = 100

all_gen_diversity = []
all_gen_multimodality = []
all_ground_truth_diversity = []
all_ground_truth_multimodality = []
all_fid = []

purpose = "valid"
step_size = [0.1, 0.2, 0.2]
dances_per_tile = [3, 3, 1]
density_thresh = [0.8, 0.75, 0.75]
############################################################

if purpose == "valid":
    labelled_data = labelled_data_valid
    labels = labels_valid
if purpose == "train":
    labelled_data = labelled_data_train
    labels = labels_train
if purpose == "test":
    labelled_data = labelled_data_test
    labels = labels_test

# encode training data and see where the average latent variable is situated for each effort
z_0 = []
z_1 = []
z_2 = []
batch = 0
onehot_encoder = utils.make_onehot_encoder(default_config.label_dim)
for i_batch, (x, y) in enumerate(zip(labelled_data, labels)):

    batch += 1

    x, y = Variable(x), Variable(y)
    x, y = x.to(default_config.device), y.to(default_config.device)

    y_label = torch.squeeze(y).item()
    batch_one_hot = utils.batch_one_hot(y, default_config.label_dim)
    y = batch_one_hot.to(default_config.device)

    z, z_mu, z_logvar = model.encode(x, y)

    if y_label == 0:
        z_0.append(z.cpu().data.numpy())
    if y_label == 1:
        z_1.append(z.cpu().data.numpy())
    if y_label == 2:
        z_2.append(z.cpu().data.numpy())

z_0 = np.squeeze(np.array(z_0))
z_1 = np.squeeze(np.array(z_1))
z_2 = np.squeeze(np.array(z_2))

z_0_center = np.mean(z_0, axis=0)
z_1_center = np.mean(z_1, axis=0)
z_2_center = np.mean(z_2, axis=0)

pca0 = PCA(n_components=2).fit(z_0 - z_0_center)
z_0_transf = pca0.transform(z_0 - z_0_center)

pca1 = PCA(n_components=2).fit(z_1 - z_1_center)
z_1_transf = pca1.transform(z_1 - z_1_center)

pca2 = PCA(n_components=2).fit(z_2 - z_2_center)
z_2_transf = pca2.transform(z_2 - z_2_center)

all_z = [z_0_transf, z_1_transf, z_2_transf]

z_min, z_max = -8, 8

grid_xs = [
    np.arange(z_min, z_max, step_size[0]),
    np.arange(z_min, z_max, step_size[1]),
    np.arange(z_min, z_max, step_size[2]),
]
grid_ys = [
    -np.arange(z_min, z_max, step_size[0]),
    -np.arange(z_min, z_max, step_size[1]),
    -np.arange(z_min, z_max, step_size[2]),
]
n_xs = len(grid_xs[0])
n_ys = len(grid_ys[0])


count = np.zeros((default_config.label_dim, n_xs, n_ys))

for y in range(default_config.label_dim):
    for i, y_coord in enumerate(grid_ys[y]):
        for j, x_coord in enumerate(grid_xs[y]):
            for z in all_z[y]:
                if x_coord < z[0] and z[0] < (x_coord + step_size[y]):
                    if y_coord < z[1] and z[1] < (y_coord + step_size[y]):
                        count[y, i, j] += 1


sum_of_counts = np.sum(count, axis=0)
density = [count[i] / sum_of_counts for i in range(default_config.label_dim)]
print(sum_of_counts)
print(np.sum(sum_of_counts))

# create array of high density tiles that we can sample from later
high_0 = []
high_1 = []
high_2 = []
all_high = []
for y in range(default_config.label_dim):
    high = []
    for i, y_coord in enumerate(grid_ys[y]):
        for j, x_coord in enumerate(grid_xs[y]):
            if (
                density[y][i, j] > density_thresh[y]
                and count[y, i, j] > dances_per_tile[y]
            ):
                high.append((x_coord, y_coord))
    all_high.append(high)

print("ALL HIGH")
print(all_high)

for i in range(stat_sampling_size):
    # MAKE DANCE
    num_gen_lab = 152
    blind_sequences = []
    for y in range(default_config.label_dim):
        one_label_seq = []
        for i in range(num_gen_lab):
            # decode within high density tile
            tile_to_pick = np.random.randint(0, len(all_high[y]))
            tile = all_high[y][tile_to_pick]
            zx = np.random.uniform(tile[0], tile[0] + step_size[y])
            zy = np.random.uniform(tile[1], tile[1] + step_size[y])
            z = np.array((zx, zy))
            z = pca0.inverse_transform(z) + z_0_center

            z_within_tile = (
                torch.tensor(z).reshape(1, -1).to(default_config.device).float()
            )

            x_create = model.sample(
                z_within_tile,
                onehot_encoder(0).reshape((1, 3)).to(default_config.device),
            )
            x_create_formatted = x_create[0].reshape((default_config.seq_len, -1, 3))

            one_label_seq = np.append(
                one_label_seq, x_create_formatted.cpu().data.numpy()
            )  # shape [n_test, 40, 53, 3]

        blind_sequences = np.append(blind_sequences, one_label_seq)

        blind_sequences = np.array(blind_sequences).reshape(
            -1, default_config.seq_len, 53, 3
        )

        # make array of original labels with size [300, 1]
        associated_labels = np.concatenate(
            (np.zeros(num_gen_lab), np.ones(num_gen_lab), 2 * np.ones(num_gen_lab))
        )

        # reshuffle sequences and labels along the first axis
        shuffler = np.random.permutation(len(associated_labels))
        associated_labels_shuffled = np.array(associated_labels)  # [shuffler])
        set_of_blind_sequences_shuffled = np.array(blind_sequences)  # [shuffler])

    # LABELS
    gen_labels = torch.tensor(
        np.concatenate(
            (
                np.zeros(num_gen_lab, dtype=int),
                np.ones(num_gen_lab, dtype=int),
                2 * np.ones(num_gen_lab, dtype=int),
            )
        )
    ).to(evaluation_device)
    ground_truth_labels = np.squeeze(labels.dataset).astype("int")

    # ACTIVATIONS
    _, gen_activations = fid_classifier_model.forward(
        torch.tensor(set_of_blind_sequences_shuffled).to(default_config.device)
    )
    _, ground_truth_activations = fid_classifier_model.forward(
        torch.tensor(labelled_data.dataset).to(evaluation_device)
    )

    # STATISTICS
    gen_statistics = metrics.calculate_activation_statistics(
        gen_activations.cpu().detach().numpy()
    )
    ground_truth_statistics = metrics.calculate_activation_statistics(
        ground_truth_activations.cpu().detach().numpy()
    )

    # FID
    fid = metrics.calculate_frechet_distance(
        ground_truth_statistics[0],
        ground_truth_statistics[1],
        gen_statistics[0],
        gen_statistics[1],
    )

    # GEN AND MULTIMOD
    gen_diversity, gen_multimodality = metrics.calculate_diversity_multimodality(
        gen_activations,
        torch.tensor(gen_labels).cpu().detach().numpy(),
        classifier_config.label_dim,
    )
    (
        ground_truth_diversity,
        ground_truth_multimodality,
    ) = metrics.calculate_diversity_multimodality(
        ground_truth_activations,
        torch.tensor(ground_truth_labels).cpu().detach().numpy(),
        classifier_config.label_dim,
    )

    all_gen_diversity.append(gen_diversity.cpu().data.numpy())
    all_gen_multimodality.append(gen_multimodality.cpu().data.numpy())
    all_ground_truth_diversity.append(ground_truth_diversity.cpu().data.numpy())
    all_ground_truth_multimodality.append(ground_truth_multimodality.cpu().data.numpy())
    all_fid.append(fid)

gen_diversity = np.mean(np.array(all_gen_diversity))
gen_multimodality = np.mean(np.array(all_gen_multimodality))
ground_truth_diversity = np.mean(np.array(all_ground_truth_diversity))
ground_truth_multimodality = np.mean(np.array(all_ground_truth_multimodality))
fid = np.mean(np.array(all_fid))

vargen_diversity = np.sqrt(np.var(np.array(all_gen_diversity)))
vargen_multimodality = np.sqrt(np.var(np.array(all_gen_multimodality)))
varground_truth_diversity = np.sqrt(np.var(np.array(all_ground_truth_diversity)))
varground_truth_multimodality = np.sqrt(
    np.var(np.array(all_ground_truth_multimodality))
)
varfid = np.sqrt(np.var(np.array(all_fid)))

####################################################

# Compute average joint distance for validation data
ajd_valid = metrics.ajd(
    model,
    evaluation_device,
    labelled_data_valid,
    labels_valid,
    default_config.label_dim,
)
ajd_train = metrics.ajd(
    model,
    evaluation_device,
    labelled_data_train,
    labels_train,
    default_config.label_dim,
)
ajd_test = metrics.ajd_test(
    model, evaluation_device, labelled_data_test, labels_test, default_config.label_dim
)

# # Plot latent space via PCA
x = torch.tensor(labelled_data_train.dataset).to(evaluation_device)
y = torch.tensor(labels_train.dataset).to(evaluation_device)
batch_one_hot = utils.batch_one_hot(y, default_config.label_dim)
y = batch_one_hot.to(evaluation_device)
y_for_encoder = y.repeat((1, default_config.seq_len, 1))
y_for_encoder = 0.33 * torch.ones_like(y_for_encoder).to(evaluation_device)
z, _, _ = model.encoder(torch.cat([x, y_for_encoder], dim=2).float())
index = np.arange(0, len(y), 1.0)
pca = PCA(n_components=10).fit(z.cpu().detach().numpy())
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(pca.explained_variance_ratio_)
axs.set_xlabel("Number of Principal Components")
axs.set_ylabel("Explained variance")
z_transformed = pca.transform(z.cpu().data.numpy())

plt.savefig(
    f"evaluate/log_files/latent_space_{default_config.load_from_checkpoint}_PC_neighbour.png"
)

fig = plt.figure()
axs = fig.add_subplot(projection="3d")

sc = axs.scatter(
    z_transformed[:, 0],
    z_transformed[:, 1],
    z_transformed[:, 2],
    c=np.squeeze(labels_train.dataset),
    alpha=0.4,
    s=0.5,
)
lp = lambda i: plt.plot(
    [],
    [],
    [],
    color=sc.cmap(sc.norm(i)),
    ms=np.sqrt(5),
    mec="none",
    label="Laban Effort {:g}".format(i),
    ls="",
    marker="o",
)[0]
handles = [lp(i) for i in np.unique(np.squeeze(labels_train.dataset))]

plt.legend(handles=handles)
plt.savefig(
    f"evaluate/log_files/latent_space_{default_config.load_from_checkpoint}_neighbour_index.png"
)

fig, axs = plt.subplots(1, 1, figsize=(8, 8))

axs.scatter(
    z_transformed[:, 0],
    z_transformed[:, 1],
    c=index,  # np.squeeze(labels_valid.dataset),
    alpha=0.2,
    s=0.1,
)
plt.savefig(
    f"evaluate/log_files/neighb/latent_space_{default_config.load_from_checkpoint}_neighbour_effort.png"
)

# Write text file with all of the metrics

####################################################

# Measure classification accuracy
acc_valid = metrics.calc_accuracy(
    model, evaluation_device, labelled_data_valid, labels_valid
)
acc_test = metrics.calc_accuracy(
    model, evaluation_device, labelled_data_test, labels_test
)

####################################################
print(round(acc_valid * 100, 1))
# Log everything
row_in_table = f"P\%   & {round(acc_valid*100,1)}\%   & {round(0*100,1)}\%  & {round(gen_diversity,1)}  & {round(gen_multimodality,1)}   & {round(ajd_test*100,1)}\% & D\% & -- "
evaluate_file = f"evaluate/log_files/neighb/evaluation_{default_config.load_from_checkpoint}_{purpose}.txt"
with open(evaluate_file, "w") as f:
    f.write(
        f"========== Metrics for checkpoint {default_config.load_from_checkpoint} ========== \n"
    )
    f.write(f"Classif Accuracy (Valid): {acc_valid} \n")
    f.write(f"Classif Accuracy (Test): {acc_test} \n")
    f.write(f"------------------------------ \n")
    f.write(f"Stats on {purpose} data \n")

    f.write(f"FID: {fid} +/- {varfid} \n")
    f.write(f"Diversity: {ground_truth_diversity} +/- {varground_truth_diversity}\n")
    f.write(
        f"Multimodality: {ground_truth_multimodality} +/- {varground_truth_multimodality} \n"
    )
    f.write(f"------------------------------ \n")
    f.write(f"Gen Diversity: {gen_diversity} +/- {vargen_diversity} \n")
    f.write(f"Gen Multimodality: {gen_multimodality} +/- {vargen_multimodality} \n")
    f.write(f"------------------------------ \n")
    f.write(f"AJD valid (recon loss): {ajd_valid}\n")
    f.write(f"AJD train (recon loss): {ajd_train}\n")
    f.write(f"AJD test (recon loss): {ajd_test}")
    f.write(f"------------------------------ \n")
    f.write(f"Row in table: \n")
    f.write(f"{row_in_table}\n")
    f.write(f"------------------------------ \n")
    f.write(
        f"Total amount of sequences (train, valid, test): {labelled_data_train.dataset.shape}, {labelled_data_valid.dataset.shape}, {labelled_data_test.dataset.shape}"
    )
