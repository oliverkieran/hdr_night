#!/bin/bash
# Basic range with steps for loop
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
for value in {100000..150000..5000}
do
python -u test_model.py test_dir=test_images/ result_dir=results/full_res_test_results/ model_dir=models/punet/learning_R_0.0001_RI_100000_NI_3/ restore_iter=$value num_input_images=3
done
echo All done

