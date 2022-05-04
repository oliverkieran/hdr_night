#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

# Basic range with steps for loop
source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
echo run_id = dped_night_loss_50000_color
for value in {52000..82000..5000}
do
echo iteration: $value
python -u test_model.py dped_dir=../../dataset_raw/ test_dir=test_images/MTK_RAW iteration=$value run_id=dped_night_loss_50000_color input_images=3
done
echo All done

