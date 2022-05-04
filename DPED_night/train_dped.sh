#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
echo "Color Adversial Images"
python -u train_model.py batch_size=28 train_size=2000 learning_rate=1e-5 num_train_iters=60000 restore_iter=33000 loss=dped_night run_id=dped_night_loss_color from_model=dped_night_loss_color input_images=3
echo All Done
