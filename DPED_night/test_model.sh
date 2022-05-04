#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
echo input_images=3
python -u test_model.py dped_dir=../../dataset_raw/ test_dir=png_images/MTK_RAW iteration=33000 run_id=dped_night_loss_color input_images=3
echo All Done
