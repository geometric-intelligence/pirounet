#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1

sweep_config = {
"name" : "my-sweep",
"method" : “bayes”,
“metric”: {
“goal”: “minimize”,
“name”:”valid_loss”
},
"parameters" : {
"epochs" : {
"values" : [50, 150]
},
"learning_rate" :{
"min": 0.00001,
"max": 0.001
},
“batchsize” : {
"values" : [8, 20, 100]
}
},
“program”: “main.py”
}
sweep_id = wandb.sweep(sweep_config)
srun --gres=gpu:1 wandb.agent(sweep_id)
