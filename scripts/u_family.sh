#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 16

python ../src/unit_exp.py --data_id family_tree --model_id standard_transformer
python ../src/unit_exp.py --data_id family_tree --model_id H_transformer
python ../src/unit_exp.py --data_id family_tree --model_id standard_MLP
python ../src/unit_exp.py --data_id family_tree --model_id H_MLP

