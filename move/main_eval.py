"""Run this file to obtain all qualitative and quantitative metrics from a run.
Specify desired metrics in eval_config.py.
"""
import os
from os.path import exists

import classifier_config
import datasets
import default_config
import eval_config
import evaluate.confusion_plot as confusion_plot
import evaluate.generate_f as generate_f
import evaluate.metrics as metrics
import models.classifiers as classifiers
import models.dgm_lstm_vae as dgm_lstm_vae
import numpy as np
import torch

filepath_to_results = os.path.join(
    os.path.abspath(os.getcwd()),
    "results/" + eval_config.load_from_checkpoint,
)

if exists(filepath_to_results) is False:
    os.makedirs(os.path.join(filepath_to_results, "quali_generation_metrics"))
    os.makedirs(os.path.join(filepath_to_results, "quali_recon_metrics"))
    os.makedirs(os.path.join(filepath_to_results, "quanti_gen_recon_metrics"))
    os.makedirs(os.path.join(filepath_to_results, "test_entanglement"))
    os.makedirs(os.path.join(filepath_to_results, "to_be_labeled"))
    os.makedirs(os.path.join(filepath_to_results, "latent_space"))


fid_classifier_model = classifiers.FID_LinearClassifier(
    input_dim=classifier_config.input_dim,
    h_dim=classifier_config.h_dim_class,
    label_dim=classifier_config.label_dim,
    seq_len=classifier_config.seq_len,
    neg_slope=classifier_config.neg_slope_classif,
    n_layers=classifier_config.n_layers_class,
).to(eval_config.evaluation_device)

classif_checkpoint_filepath = os.path.join(
    os.path.abspath(os.getcwd()),
    "saved/classifier/" + classifier_config.load_from_checkpoint + ".pt",
)
classif_checkpoint = torch.load(classif_checkpoint_filepath)
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
).to(eval_config.evaluation_device)

dgm_checkpoint_filepath = os.path.join(
    os.path.abspath(os.getcwd()), "saved/" + default_config.load_from_checkpoint + ".pt"
)

checkpoint = torch.load(dgm_checkpoint_filepath)
model.load_state_dict(checkpoint["model_state_dict"])
latest_epoch = checkpoint["epoch"]

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


if eval_config.quali_generation_metrics:

    generate_f.generate_and_save(
        model=model,
        config=eval_config,
        epoch=latest_epoch,
        num_artifacts=eval_config.num_random_artifacts,
        type="random",
        results_path=filepath_to_results
        + "/quali_generation_metrics/random_generation",
        comic=False,
    )

    generate_f.generate_and_save(
        model=model,
        config=eval_config,
        epoch=latest_epoch,
        num_artifacts=eval_config.num_cond_artifacts_per_lab,
        type="cond",
        encoded_data=labelled_data_train,
        encoded_labels=labels_train,
        results_path=filepath_to_results
        + "/quali_generation_metrics/random_generation",
        comic=False,
    )

if eval_config.quali_recon_metrics:

    generate_f.reconstruct(
        model,
        config=eval_config,
        epoch=latest_epoch,
        input_data=labelled_data_test,
        input_label=labels_test,
        purpose="test",
        results_path=filepath_to_results + "/quali_recon_metrics/test_recon",
        comic=False,
    )

    generate_f.reconstruct(
        model,
        config=eval_config,
        epoch=latest_epoch,
        input_data=labelled_data_valid,
        input_label=labels_valid,
        purpose="valid",
        results_path=filepath_to_results + "/quali_recon_metrics/valid_recon",
        comic=False,
    )

if eval_config.test_entanglement:

    generate_f.generate_and_save_one_move(
        model, config=eval_config, path=filepath_to_results + "/test_entanglement"
    )

    generate_f.plot_dist_one_move(
        model,
        config=eval_config,
        path=filepath_to_results + "/test_entanglement",
        n_one_moves=30,
    )

if eval_config.generate_for_blind_labeling:

    gen_dance, gen_label = generate_f.generate_cond(
        model,
        config=eval_config,
        num_gen_cond_lab=100,
        encoded_data=labelled_data_train,
        encoded_labels=labels_train,
        shuffle=True,
    )
    path_blind_generation = filepath_to_results + "/to_be_labeled"
    np.save(
        f"{path_blind_generation}/dance_{eval_config.load_from_checkpoint}", gen_dance
    )
    np.save(
        f"{path_blind_generation}/labels_{eval_config.load_from_checkpoint}", gen_label
    )

if eval_config.plot_recognition_accuracy:

    confusion_plot.plot_recognition_accuracy(
        human_labels=eval_config.human_labels,
        pirounet_labels=eval_config.pirounet_labels,
        path=filepath_to_results + "/quanti_gen_recon_metrics",
    )

if eval_config.plot_classification_accuracy:

    confusion_plot.plot_classification_accuracy(
        model,
        eval_config,
        labelled_data_valid,
        labels_valid,
        purpose="valid",
        path=filepath_to_results + "/quanti_gen_recon_metrics",
    )

    confusion_plot.plot_classification_accuracy(
        model,
        eval_config,
        labelled_data_test,
        labels_test,
        purpose="test",
        path=filepath_to_results + "/quanti_gen_recon_metrics",
    )

    confusion_plot.plot_classification_accuracy(
        model,
        eval_config,
        labelled_data_train,
        labels_train,
        purpose="train",
        path=filepath_to_results + "/quanti_gen_recon_metrics",
    )

if eval_config.plot_latent_space:

    generate_f.plot_latentspace(
        model,
        eval_config,
        labelled_data_train,
        labels_train,
        path=filepath_to_results + "/latent_space",
    )

if eval_config.quanti_gen_recon_metrics:

    # BOOTSTRAP LOOP TO TAKE DISTRIBUTION ON STATS
    all_gen_diversity = []
    all_gen_multimodality = []
    all_ground_truth_diversity = []
    all_ground_truth_multimodality = []
    all_fid = []
    for i in range(eval_config.stat_sampling_size):
        # MAKE DANCE
        gen_dance, gen_labels = generate_f.generate_cond(
            model,
            eval_config.label_dim,
            eval_config.num_gen_cond_lab,
            encoded_data=labelled_data_train,
            encoded_labels=labels_train,
            shuffle=True,
        )

        # LABELS
        gen_labels = gen_labels.to(eval_config.evaluation_device)
        ground_truth_labels = np.squeeze(labels_train.dataset).astype("int")

        # ACTIVATIONS
        _, gen_activations = fid_classifier_model.forward(
            torch.tensor(gen_dance).to(eval_config.device)
        )
        _, ground_truth_activations = fid_classifier_model.forward(
            torch.tensor(labelled_data_train.dataset).to(eval_config.evaluation_device)
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
            eval_config.label_dim,
        )
        (
            ground_truth_diversity,
            ground_truth_multimodality,
        ) = metrics.calculate_diversity_multimodality(
            ground_truth_activations,
            torch.tensor(ground_truth_labels).cpu().detach().numpy(),
            eval_config.label_dim,
        )

        all_gen_diversity.append(gen_diversity.cpu().data.numpy())
        all_gen_multimodality.append(gen_multimodality.cpu().data.numpy())
        all_ground_truth_diversity.append(ground_truth_diversity.cpu().data.numpy())
        all_ground_truth_multimodality.append(
            ground_truth_multimodality.cpu().data.numpy()
        )
        all_fid.append(fid)

    gen_diversity = np.mean(np.array(all_gen_diversity))
    gen_multimodality = np.mean(np.array(all_gen_multimodality))
    ground_truth_diversity = np.mean(np.array(all_ground_truth_diversity))
    ground_truth_multimodality = np.mean(np.array(all_ground_truth_multimodality))
    fid = np.mean(np.array(all_fid))

    siggen_diversity = np.sqrt(np.var(np.array(all_gen_diversity)))
    siggen_multimodality = np.sqrt(np.var(np.array(all_gen_multimodality)))
    sigground_truth_diversity = np.sqrt(np.var(np.array(all_ground_truth_diversity)))
    sigground_truth_multimodality = np.sqrt(
        np.var(np.array(all_ground_truth_multimodality))
    )
    sigfid = np.sqrt(np.var(np.array(all_fid)))

    # AJD
    ajd_valid = metrics.ajd(
        model,
        eval_config.evaluation_device,
        labelled_data_valid,
        labels_valid,
        eval_config.label_dim,
    )
    ajd_train = metrics.ajd(
        model,
        eval_config.evaluation_device,
        labelled_data_train,
        labels_train,
        eval_config.label_dim,
    )
    ajd_test = metrics.ajd_test(
        model,
        eval_config.evaluation_device,
        labelled_data_test,
        labels_test,
        eval_config.label_dim,
    )

    # ACCURACY
    acc_valid = metrics.calc_accuracy(
        model, eval_config.evaluation_device, labelled_data_valid, labels_valid
    )
    acc_test = metrics.calc_accuracy(
        model, eval_config.evaluation_device, labelled_data_test, labels_test
    )

    ####################################################
    # Log everything
    row_in_table = f"P\%   & {round(acc_valid*100,1)}\%   & {round(0*100,1)}\%  & {round(gen_diversity,1)}  & {round(gen_multimodality,1)}   & {round(ajd_test*100,1)}\% & D\% & -- "
    evaluate_file = (
        filepath_to_results
        + f"/quanti_gen_recon_metrics/logfile_{eval_config.load_from_checkpoint}.txt"
    )
    with open(evaluate_file, "w") as f:
        f.write(
            f"========== Metrics for checkpoint {eval_config.load_from_checkpoint} ========== \n"
        )
        f.write(f"Classif Accuracy (Valid): {acc_valid} \n")
        f.write(f"Classif Accuracy (Test): {acc_test} \n")
        f.write(f"------------------------------ \n")
        f.write(f"Stats \n")

        f.write(f"FID: {fid} +/- {sigfid} \n")
        f.write(
            f"Diversity: {ground_truth_diversity} +/- {sigground_truth_diversity}\n"
        )
        f.write(
            f"Multimodality: {ground_truth_multimodality} +/- {sigground_truth_multimodality} \n"
        )
        f.write(f"------------------------------ \n")
        f.write(f"Gen Diversity: {gen_diversity} +/- {siggen_diversity} \n")
        f.write(f"Gen Multimodality: {gen_multimodality} +/- {siggen_multimodality} \n")
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
