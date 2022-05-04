#!/bin/bash
# Basic range with steps for loop
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
for value in {100000..150000..5000}
do
python -u test_model.py test_dir=test_images/ result_dir=results/full_res_test_results/ model_dir=models/punet/learning_R_5e-05_RI_100000_NI_3/
done
echo All done
