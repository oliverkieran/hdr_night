#!/bin/bash
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u train_model.py model_dir=learning_R_0.0001_RI_0/ restore_iter=100000 learning_rate=0.0001 num_train_iters=50000
