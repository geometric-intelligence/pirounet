"""Default configuration parameters para entrenar PirouNet
con el corpus de movimiento de Diego (BVH → NPY, 60 loops).

Pipeline activo: get_model_specific_data()
Eje de entrenamiento: time (Eje 1 — Estado gravitacional)

INSTRUCCIONES:
  1. Copiar este archivo a pirounet/pirounet/default_config.py
  2. Completar 'project' y 'entity' con tus datos de wandb
  3. Asegurarse de que en pirounet/pirounet/data/ existan:
       - mariel_diego.npy
       - labels_from_app.csv
  4. Comentar el recorte en datasets.py (línea):
       ds = ds[500:-500, point_mask]  →  ds = ds[:, point_mask]
  5. Ejecutar: python main.py
"""

import torch

# ── Identificación del run ────────────────────────────────────────────
run_name = "pirounet_diego_v1"
load_from_checkpoint = None   # None = entrenar desde cero

# ── Wandb ─────────────────────────────────────────────────────────────
project = "pirounet-diego"     # cambia por tu nombre de proyecto en wandb
entity  = "tu_usuario_wandb"   # cambia por tu usuario de wandb

# ── Hardware ──────────────────────────────────────────────────────────
which_device = "0"
device = (
    torch.device("cuda:" + str(which_device))
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# ── Entrenamiento ─────────────────────────────────────────────────────
epochs        = 500
learning_rate = 3e-4
batch_size    = 80
with_clip     = False

# ── Datos de entrada ──────────────────────────────────────────────────
seq_len   = 40
input_dim = 159    # 53 joints × 3 coords — NO modificar
label_dim = 3      # 3 clases: Anclado / Suspendido / Ingrávido
amount_of_labels = 1

# Eje activo: "time" usa columna 1 del CSV (Eje 1 — Estado gravitacional)
# Cambiar a "space" para usar columna 2 (Eje 2 — Presencia corporal)
effort = "time"

# Pipeline: get_model_specific_data (original del paper)
# fraction_label: fracción del dataset etiquetado usada para entrenamiento
# (el 5% inicial se reserva para validación, el 3% final para test)
fraction_label = 0.789   # ~78.9% de las etiquetas van a train
shuffle_data   = False
train_ratio    = None    # None → activa get_model_specific_data
train_lab_frac = None    # None → activa get_model_specific_data

# ── Arquitectura LSTM VAE ─────────────────────────────────────────────
kl_weight   = 1
neg_slope   = 0
n_layers    = 5
h_dim       = 100
latent_dim  = 256

# ── Arquitectura clasificador ─────────────────────────────────────────
h_dim_classif    = 100
neg_slope_classif = 0
n_layers_classif  = 2
