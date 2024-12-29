#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -n 16

for ARG in $(python -c "import numpy as np; print(' '.join(map(str, np.linspace(51, 100, 20, dtype=int))))"); do
    echo "Running with seed $ARG:"
    python ../src/unit_exp.py --data_id family_tree --model_id H_MLP --seed $ARG
    echo
done