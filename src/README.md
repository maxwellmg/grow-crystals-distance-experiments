## How to add new dataset for experiments

1. Implement a function which returns the dataset dictionary in `utils/dataset.py`.
2. Choose a unique id for the new dataset. Implement a function which evaluates the quality of representation in `utils/crystal_metric.py`. Modify the function `crystal_metric` to support the new data_id.
3. Add the new data_id to the array `data_id_choices` in `run_exp.py`.
4. If any auxiliary information is required to evaluate the representations, add them to the dictionary `aux_info` in `run_exp.py`. Sometimes, these information may depend on the specific dataset; In such cases, make any necessary modifications within each of the three experiment for loops in `run_exp.py`.
5. Now, you're ready to test the new dataset! Command format is:
`python run_exp.py --data_id new_data_id --model_id H_MLP`.