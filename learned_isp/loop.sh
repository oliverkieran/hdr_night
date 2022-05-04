#!/bin/bash
# Basic range with steps for loop
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
for value in {50000..100000..5000}
do
python -u test_model.py test_dir=full_res_test/ model_dir=restore_iter=$value
done
echo All done

