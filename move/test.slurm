#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1

srun --gres=gpu:1 wandb agent mathildepapillon/move-move/925v5ka3
