#!/bin/bash
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u train_model.py restore_iter=0 learning_rate=5e-4

