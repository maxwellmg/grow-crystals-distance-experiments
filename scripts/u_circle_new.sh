#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -p tegmark
#SBATCH --gres=gpu:a100:1

for ARG in $(python -c "import numpy as np; print(' '.join(map(str, np.linspace(0, 1000, 20, dtype=int))))"); do
    echo "Running with seed $ARG:"
    python ../src/unit_exp.py --data_id circle --model_id H_transformer --seed $ARG
    echo
done