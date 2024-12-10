import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

import sys

from dataset import parallelogram_dataset, repeat_dataset, ToyDataset
from model import ToyTransformer
from visualization import visualize_embedding

import argparse

parser = argparse.ArgumentParser(description='Toy Transformer')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--d_model', type=int, default=16, help='Embedding and hidden size')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--data_size', type=int, default=1000, help='Size of the dataset')
parser.add_argument('--use_harmonic', type=int, default=0, help='Use Harmonic Loss')


args = parser.parse_args()
seed = args.seed
batch_size = args.batch_size
data_size = args.data_size
d_model = args.d_model


np.random.seed(seed)
torch.manual_seed(seed)

torch.set_default_tensor_type(torch.DoubleTensor)

p = 5
embd_dim = 5
input_token = 3
lattice_dim = 2
vocab_size = p ** lattice_dim

# data
dataset = parallelogram_dataset(p=p, dim=lattice_dim, num=data_size, seed=seed)
dataset = repeat_dataset(dataset)

# Parameters
#d_model = 16     # Embedding and hidden size
nhead = 2        # Number of attention heads
num_layers = 2   # Number of transformer layers
seq_len = 3      # Sequence length
num_epochs = 500
learning_rate = 0.001

    
toy_dataset = ToyDataset(dataset['train_data_id'], dataset['train_label'])
dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=batch_size, shuffle=True)

param_dict = {
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'dataloader': dataloader
}

print(f"Training model with d_model = {d_model}, batch_size = {batch_size}, data_size = {data_size}")

model = ToyTransformer(vocab_size, d_model, nhead, num_layers, seq_len, use_dist_layer=0 if args.use_harmonic == 0 else 1)
model.to('cuda')

model.train(param_dict)

result_dict = model.eval()

with open("../results/sweep_results.csv","a") as file:
    file.write(f"{args.use_harmonic},{data_size},{d_model},{batch_size},{result_dict['parallelogram_quality']},{result_dict['variances'][0]+result_dict['variances'][1]}\n")