#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 32

sizes=$(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(1, 4, num=10, dtype=int))))")


for size in $sizes
do
    python3 ../sweep_transformers.py --data_size $size --use_harmonic 0
done

