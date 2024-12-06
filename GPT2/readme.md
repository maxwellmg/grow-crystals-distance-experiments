# Start

* Step 0: This should immediately work on supercloud (except that perhaps you need to pip install wandb etc., I don't remember exactly).

* Step 1: create a folder data/openwebtext, and put both binary files train.bin and val.bin into the folder. Binary files can be downloaded from [here](https://www.dropbox.com/scl/fo/v24k2eltevgiszdfvean6/AF0j1Pu9ladYpDZbqSVKHGI?rlkey=jwa73nxrwt5bj13a6c9q0z20w&st=090g6v8w&dl=0).

* Step 2: in terminal type `sbatch train_adam_l2loss.sh`. That's it!

# Notice
* The code is based on [sophia repo](https://github.com/Liuhong99/Sophia/tree/main), which in turn is based on [nanogpt](https://github.com/karpathy/nanoGPT/). The training pipeline might be unnecessarily complicated for our purposes (a lot of parallelization etc.).
* My major changes (relevant to harmonic losses) are in `model_l2loss.py` and highlighted with comments "Ziming's note".
* To change configurations, e.g., the size of the network, go to  `config/train_gpt2_small_adam_l2loss.py`. Although there are some hyperparameters being set up at the beginning of `train_adam_l2loss.py`, these hyperparameters are later overwritten by `config/train_gpt2_small_adam_l2loss.py`.
