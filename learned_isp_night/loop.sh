#!/bin/bash
# Basic range with steps for loop
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
for value in {50000..100000..5000}
do
python -u test_model.py test_dir=png_images/ result_dir=results/full_res_test_results/ model_dir=models/punet/learning_R_5e-05_RI_0_NI_1/ restore_iter=$value num_input_images=1
done
echo All done

