
# Define arrays for parameters
data_ids=("circle") #  "circle"
model_ids=("H_transformer")
splits=(3)  # Modify as needed
ns=(2) # I have 3 - 10 left to do for circle

# Iterate over all combinations
for data_id in "${data_ids[@]}"; do
    for model_id in "${model_ids[@]}"; do
        for split in "${splits[@]}"; do
            for n in "${ns[@]}"; do
                PYTHONPATH=$(pwd) python src/run_exp.py --data_id "$data_id" --model_id "$model_id" --split "$split" --n "$n" > output_"$data_id"_"$n".txt 2>&1
            done
        done
    done
done
