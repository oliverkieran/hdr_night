#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u test_model.py test_dir=mediatek_raw/ model_dir=models/punet_vgg/ result_dir=results/enhanced_test_images_punet_vgg/ restore_iter=100000 img_h=256 img_w=256 use_gpu=True test_image=True

