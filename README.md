# PirouNet #

Official PyTorch implementation of the paper “PirouNet: Creating Intentional Dance with Semi-Supervised Conditional Recurrent Variational Autoencoders”
***[[Pre-print](https://arxiv.org/pdf/2207.12126.pdf)], accepted for publication at [[EAI ArtsIT 2022](https://artsit.eai-conferences.org/2022/)]***

PirouNet is a semi-supervised conditional recurrent variational autoencoder. This code is responsible for training and evaluating the model. Labels must be created separately prior to training. We propose this [dance labeling web application](https://github.com/mathildepapillon/pirounet_label) which can be customized to the user's labeling needs.

![Overview of PirouNet's LSTM+VAE architecture.](/images/arch_overview.jpeg)

## 🌎 Bibtex ##
If this code is useful to your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2207.12126,
  doi = {10.48550/ARXIV.2207.12126},

  url = {https://arxiv.org/abs/2207.12126},

  author = {Papillon, Mathilde and Pettee, Mariel and Miolane, Nina},

  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {PirouNet: Creating Intentional Dance with Semi-Supervised Conditional Recurrent Variational Autoencoders},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```


## 🏡 Installation ##

This codes runs on Python 3.8. We recommend using Anaconda for easy installation. To create the necessary conda environment, run:
```
cd pirounet
conda env create -f environment.yml
conda activate choreo
```

## 🚀 Training ##

To train a new model (see below for loading a saved model), follow the steps below.

#### 1. Set up [Wandb](https://wandb.ai/home) logging.

Wandb is a powerful tool for logging performance during training, as well as animation artifacts. To use it, simply [create an account](https://wandb.auth0.com/login?state=hKFo2SBNb0U4SjE0ZWN3OGZtbTlJWTRpYkNmU0dUTWZKSDk3Y6FupWxvZ2luo3RpZNkgODhWd254WW1zdG51RTREd0pWOGVKWVVzZkVOZ0dydGqjY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=dEZVS3dvYXFVSjdjZFFGdw%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true), then run:
```
wandb login
```
to sign into your account.

#### 2. Specify hyperparameters in default_config.py.

For wandb: Specify your wandb account under “entity” and title of the project under “project_name”. “run_name” will title this specific run within the project.

If specified, “load_from_checkpoint” indicates the saved model to load. Leave as “None” for training a new model.

Other hyperparameters are organized by category: hardware (choice of CUDA device), training, input data, LSTM VAE architecture, and classifier architecture.

#### 3. Train!
For a single run, use the command:
```
python main.py
```
For a hyperparameter sweep (multiple runs), we invite you to follow wandb’s [Quickstart guide](https://docs.wandb.ai/guides/sweeps/quickstart) and run the resulting wandb sweep command.

### 📕 Load a saved model.
There are basic types of models to load:
* *PirouNet_{watch}.*
Copy contents of saved_models/pirounet_watch_config.py file into default_config.py.

* *PirouNet_{dance}.*
Copy contents of saved_models/pirounet_dance_config.py file into default_config.py.

* *Your new model.*
In default_config.py, specify “load_from_checkpoint” as the name and epoch corresponding your new model:“checkpoint_{run_name}_epoch{epoch}”.
Make sure the rest of the hyperparameters match those you used during training.

Once this is done, there are two options:
1. Continue training using this saved model as a starting point. See “Training” section.
2. Evaluate this saved model.

## 🕺 Evaluation ##

1. Follow the “Load a saved model” instructions to configure default_config.py.
2. Specify the parameters of the evaluation in eval_config.py. Note that “plot_recognition_accuracy” should only be set to True once a human labeler has blindly labeled PirouNet-generated dance sequences (using generate_for_blind_labeling and the web-labeling app), and exported the csv of labels to the pirounet/pirounet/data directory.
3. Unzip the pre-saved classifier model in saved_models/classifier.
4. Run the command:
```
python main_eval.py
```
This will produce a subfolder in pirounet/results containing all the qualitative and quantitative metrics included in our paper, as well as extra plots of the latent space and its entanglement. Among the qualitative generation metrics, two examples are provided below.

**Conditionally created dance sequences:**
![Animated dance sequences conditionally created by PirouNet.](/images/side_by_side_pirounet_originals.gif)

**Reconstructed dance sequence:**
![PirouNet reconstructs input dance.](/images/side_by_side_recon.gif)

## 💃 Authors ##
[Mathilde Papillon](https://sites.google.com/view/mathildepapillon)

[Mariel Pettee](https://mariel-pettee.github.io/)

[Nina Miolane](https://www.ninamiolane.com/)
