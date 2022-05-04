#!/bin/bash
source /itet-stor/skaeser/net_scratch/conda/etc/profile.d/conda.sh
conda activate tencu10
python -u test_model.py "$@"
