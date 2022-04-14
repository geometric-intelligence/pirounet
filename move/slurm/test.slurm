#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00


timeout 89h srun --gres=gpu:1 python main.py
if [[ $? -eq 124 ]]; then
  sbatch test.slurm
fi