#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
#python -u test_model.py model=mediatek_fullres test_subset=full iteration=49000
python -u test_model.py test_dir=test_images/ iteration=30000 run_id=pynet_loss resolution=phone
