#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u train_model.py batch_size=32 train_size=5000 learning_rate=1e-5 num_train_iters=40000 restore_iter=30000 loss=pynet run_id=pynet_loss
