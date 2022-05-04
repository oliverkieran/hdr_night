#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u train_model_generative_only.py batch_size=32 train_size=5000 learning_rate=1e-4 num_train_iters=20000 restore_iter=0 run_id=pynet_color_loss

