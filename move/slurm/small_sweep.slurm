#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00


srun --gres=gpu:1 wandb agent mathildepapillon/move-move/tld50t54