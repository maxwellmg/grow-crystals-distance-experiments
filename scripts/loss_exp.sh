#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 16

# Define arrays for parameters
data_ids=("circle") #  "circle"
model_ids=("H_MLP")
splits=(3)  # Modify as needed
ns=(1)

# Iterate over all combinations
for data_id in "${data_ids[@]}"; do
    for model_id in "${model_ids[@]}"; do
        for split in "${splits[@]}"; do
            for n in "${ns[@]}"; do
                PYTHONPATH=$(pwd) python src/run_exp.py --data_id "$data_id" --model_id "$model_id" --split "$split" --n "$n" > output_softnn_"$data_id"_"$model_id".txt 2>&1  
            done
        done
    done
done