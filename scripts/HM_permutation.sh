#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:1

python ../src/run_exp.py --data_id permutation --model_id H_MLP
