"""Run this script to get all the quantitative metrics for
a specific model, except danceability and recognition accuracy. 
These require independent labeling of generated dance.

Before running, in default_config_eval.py:
1. Choose DGM model checkpoint.
2. Choose classifier model checkpoint.
3. Make sure params in default_config_eval.py match those of 
the models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

import models.classifiers as classifiers
import models.dgm_lstm_vae as dgm_lstm_vae
import classifier_config
import default_config
import datasets
import evaluate.generate_f as generate_f
import models.utils as utils
import evaluate.metrics as metrics

evaluation_device = (
    torch.device("cuda:"+str(classifier_config.which_device)) if torch.cuda.is_available() else torch.device("cpu")
)

fid_classifier_model = classifiers.LinearClassifier(
    input_dim=classifier_config.input_dim,
    h_dim=classifier_config.h_dim_class,
    label_dim=classifier_config.label_dim,
    seq_len=classifier_config.seq_len,
    neg_slope=classifier_config.neg_slope_classif,
    n_layers=classifier_config.n_layers_class,
).to(evaluation_device)

classif_old_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), "saved/classifier/" + classifier_config.load_from_checkpoint + ".pt")
classif_checkpoint = torch.load(classif_old_checkpoint_filepath)
fid_classifier_model.load_state_dict(classif_checkpoint['model_state_dict'])

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
).to(evaluation_device)

old_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), "saved/" + default_config.load_from_checkpoint + ".pt")
checkpoint = torch.load(old_checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])

(
labelled_data_train, 
labels_train, 
_, 
labelled_data_valid, 
labels_valid, 
labelled_data_test, 
labels_test,
_
) = datasets.get_model_data(classifier_config)

####################################################

# BOOTSTRAP LOOP TO TAKE DISTRIBUTION ON STATS

stat_sampling_size = 100

all_gen_diversity = []
all_gen_multimodality = []
all_ground_truth_diversity = []
all_ground_truth_multimodality = []
all_fid = []
for i in range(stat_sampling_size):
    # MAKE DANCE
    num_gen_per_lab = 152
    gen_dance, gen_labels = generate_f.generate_many(model, classifier_config.label_dim, num_gen_per_lab, classifier_config.seq_len)
    ground_truth_labels = np.squeeze(labels_valid.dataset).astype('int')

    # ACTIVATIONS
    _, gen_activations = fid_classifier_model.forward(gen_dance.to(evaluation_device))
    _, ground_truth_activations = fid_classifier_model.forward(
        torch.tensor(labelled_data_valid.dataset).to(evaluation_device))

    # STATISTICS
    gen_statistics = metrics.calculate_activation_statistics(gen_activations.cpu().detach().numpy())
    ground_truth_statistics = metrics.calculate_activation_statistics(ground_truth_activations.cpu().detach().numpy())

    # FID
    fid = metrics.calculate_frechet_distance(
        ground_truth_statistics[0], 
        ground_truth_statistics[1],
        gen_statistics[0], 
        gen_statistics[1]
        )

    # GEN AND MULTIMOD
    gen_diversity, gen_multimodality = metrics.calculate_diversity_multimodality(
        gen_activations, 
        torch.tensor(gen_labels).cpu().detach().numpy(), 
        classifier_config.label_dim)
    ground_truth_diversity, ground_truth_multimodality = metrics.calculate_diversity_multimodality(
        ground_truth_activations, 
        torch.tensor(ground_truth_labels).cpu().detach().numpy(), 
        classifier_config.label_dim)
    
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

siggen_diversity = np.sqrt(np.var(np.array(all_gen_diversity)))
siggen_multimodality = np.sqrt(np.var(np.array(all_gen_multimodality)))
sigground_truth_diversity = np.sqrt(np.var(np.array(all_ground_truth_diversity)))
sigground_truth_multimodality = np.sqrt(np.var(np.array(all_ground_truth_multimodality)))
sigfid = np.sqrt(np.var(np.array(all_fid)))

####################################################

# Compute average joint distance for validation data
ajd_valid = metrics.ajd(model, evaluation_device, labelled_data_valid, labels_valid, default_config.label_dim)
ajd_train = metrics.ajd(model, evaluation_device, labelled_data_train, labels_train, default_config.label_dim)
ajd_test = metrics.ajd_test(model, evaluation_device, labelled_data_test, labels_test, default_config.label_dim)

# # Plot latent space via PCA
x = torch.tensor(labelled_data_train.dataset).to(evaluation_device)
y = torch.tensor(labels_train.dataset).to(evaluation_device)
batch_one_hot = utils.batch_one_hot(y, default_config.label_dim)
y = batch_one_hot.to(evaluation_device)
y_for_encoder = y.repeat((1, default_config.seq_len, 1))
y_for_encoder = 0.33 * torch.ones_like(y_for_encoder).to(evaluation_device)
z, _, _ = model.encoder(torch.cat([x, y_for_encoder], dim=2).float())
index = np.arange(0, len(y), 1.)
pca = PCA(n_components=10).fit(z.cpu().detach().numpy())
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(pca.explained_variance_ratio_)
axs.set_xlabel("Number of Principal Components")
axs.set_ylabel("Explained variance")
z_transformed = pca.transform(z.cpu().data.numpy())

plt.savefig(f'evaluate/log_files/latent_space_{default_config.load_from_checkpoint}_PC_nolab_train.png')

fig = plt.figure()
axs = fig.add_subplot(projection='3d')

sc = axs.scatter(
        z_transformed[:, 0],
        z_transformed[:, 1],
        z_transformed[:, 2],
        c=np.squeeze(labels_train.dataset),
        alpha=0.4,
        s=0.5
        
    )
lp = lambda i: plt.plot([],[],[],color=sc.cmap(sc.norm(i)), ms=np.sqrt(5), mec="none",
                        label="Laban Effort {:g}".format(i), ls="", marker="o")[0]
handles = [lp(i) for i in np.unique(np.squeeze(labels_train.dataset))]

plt.legend(handles=handles)
plt.savefig(f'evaluate/log_files/latent_space_{default_config.load_from_checkpoint}_indexednolab_train.png')

fig, axs = plt.subplots(1, 1, figsize=(8, 8))

axs.scatter(
        z_transformed[:, 0],
        z_transformed[:, 1],
        c=index, #np.squeeze(labels_valid.dataset),
        alpha=0.2,
        s=0.1
    )
plt.savefig(f'evaluate/log_files/latent_space_{default_config.load_from_checkpoint}_effortnolab_train.png')

# Write text file with all of the metrics

####################################################

# Measure classification accuracy
acc_valid = metrics.calc_accuracy(model, evaluation_device, labelled_data_valid, labels_valid)
acc_test = metrics.calc_accuracy(model, evaluation_device, labelled_data_test, labels_test)

####################################################
# Log everything
row_in_table = f'P\%   & {round(acc_valid*100,1)}\%   & {round(0*100,1)}\%  & {round(gen_diversity,1)}  & {round(gen_multimodality,1)}   & {round(ajd_test*100,1)}\% & D\% & -- '
evaluate_file = f'evaluate/log_files/arch_impact/evaluation_{default_config.load_from_checkpoint}.txt'
with open(evaluate_file, 'w') as f:
    f.write(
        f'========== Metrics for checkpoint {default_config.load_from_checkpoint} ========== \n'
    )
    f.write(f'Classif Accuracy (Valid): {acc_valid} \n')
    f.write(f'Classif Accuracy (Test): {acc_test} \n')
    f.write(f'------------------------------ \n')
    f.write(f'FID: {fid} +/- {sigfid} \n')
    f.write(f'Diversity: {ground_truth_diversity} +/- {sigground_truth_diversity}\n')
    f.write(f'Multimodality: {ground_truth_multimodality} +/- {sigground_truth_multimodality} \n')
    f.write(f'------------------------------ \n')
    f.write(f'Gen Diversity: {gen_diversity} +/- {siggen_diversity} \n')
    f.write(f'Gen Multimodality: {gen_multimodality} +/- {siggen_multimodality} \n')
    f.write(f'------------------------------ \n')
    f.write(f'AJD valid (recon loss): {ajd_valid}\n')
    f.write(f'AJD train (recon loss): {ajd_train}\n')
    f.write(f'AJD test (recon loss): {ajd_test}')
    f.write(f'------------------------------ \n')
    f.write(f'Row in table: \n')
    f.write(f'{row_in_table}\n')
    f.write(f'------------------------------ \n')
    f.write(f'Total amount of sequences (train, valid, test): {labelled_data_train.dataset.shape}, {labelled_data_valid.dataset.shape}, {labelled_data_test.dataset.shape}')
