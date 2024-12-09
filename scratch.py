#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset import parallelogram_dataset, repeat_dataset
from model import DistLayer

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

torch.set_default_tensor_type(torch.DoubleTensor)

p = 5
embd_dim = 5
input_token = 3
lattice_dim = 2
vocab_size = p ** lattice_dim

    
# data
dataset = parallelogram_dataset(p=p, dim=lattice_dim, num=1000, seed=seed)
dataset = repeat_dataset(dataset)

# Parameters
d_model = 16     # Embedding and hidden size
nhead = 2        # Number of attention heads
num_layers = 2   # Number of transformer layers
seq_len = 3      # Sequence length
num_epochs = 500
batch_size = 16
learning_rate = 0.001


# Dataset and DataLoader
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
toy_dataset = ToyDataset(dataset['train_data_id'], dataset['train_label'])
print(vocab_size, dataset['vocab_size'])

dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=batch_size, shuffle=True)

# 2-Layer Transformer Model with Explicit Residual Connections
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(ToyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Define transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dist = DistLayer(d_model, d_model, n=1., eps=1e-4, bias=False)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding

        # Pass through transformer layers with residual connections
        x = embedded
        for layer in self.layers:
            x = layer(x,x) + x  # Explicit residual connection

        logits = self.fc(x[:, -1])  # Only predict the last token
        return logits

model = ToyTransformer(vocab_size, d_model, nhead, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        logits = model(batch_inputs)
        batch_indices = torch.arange(logits.size(0))
        loss = ((1/(logits[batch_indices,batch_targets] + 1e-4)) / (1/(logits + 1e-4)).sum(dim=1)).sum()
#        loss = criterion(logits, batch_targets.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# %%
emb = model.embedding.weight

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
emb_pca = pca.fit_transform(emb.detach().numpy())

for i in range(len(emb_pca)):
    plt.text(emb_pca[i, 0], emb_pca[i, 1], str(i), fontsize=12)
    plt.scatter(emb_pca[:, 0], emb_pca[:, 1])
# %%
