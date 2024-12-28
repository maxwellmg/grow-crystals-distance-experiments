#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:a100:1

python ../src/run_exp.py --data_id equivalence --model_id standard_MLP

