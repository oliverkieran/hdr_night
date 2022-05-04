#!/bin/bash
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u train_model.py model_dir=learning_R_0.0001_RI_0_NI_3/ restore_iter=100000 learning_rate=5e-5 num_train_iters=50000 num_input_images=3


