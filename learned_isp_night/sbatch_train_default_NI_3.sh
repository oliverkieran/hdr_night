#!/bin/bash
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u train_model.py learning_rate=0.00005 restore_iter=0 num_train_iters=100000 num_input_images=3


