#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 16

python ../src/run_exp.py --data_id family_tree --model_id H_MLP

