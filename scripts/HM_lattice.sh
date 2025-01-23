#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:1

python ../src/run_exp.py --data_id lattice --model_id H_MLP
python ../src/run_exp.py --data_id family_tree --model_id H_MLP
python ../src/run_exp.py --data_id equivalence --model_id H_MLP
python ../src/run_exp.py --data_id circle --model_id H_MLP
