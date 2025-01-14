#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 16

# Define arrays for parameters
data_ids=("lattice") #  "circle"
model_ids=("standard_transformer" "H_transformer" "standard_MLP" "H_MLP")
splits=(1)  # Modify as needed
ns=(1)

# Iterate over all combinations
for data_id in "${data_ids[@]}"; do
    for model_id in "${model_ids[@]}"; do
        for split in "${splits[@]}"; do
            for n in "${ns[@]}"; do
                python ../src/run_exp.py --data_id "$data_id" --model_id "$model_id" --split "$split" --n "$n"
            done
        done
    done
done
