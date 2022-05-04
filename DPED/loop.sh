#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

# Basic range with steps for loop
source /itet-stor/ollehman/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
for value in {28000..30000..1000}
do
python -u test_model.py iteration=$value run_id=pynet_loss
done
echo All done

