import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import math

from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., unembd=False, weight_tied=False, seed=0):
        super(MLP, self).__init__()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            linear_list.append(nn.Linear(shp[i], shp[i+1]))
        
        self.embedding = torch.nn.Parameter(torch.normal(0,1/torch.tensor(embd_dim),size=(vocab_size, embd_dim))*init_scale)
        #self.embedding = torch.nn.Parameter(torch.normal(0,1,size=(vocab_size, embd_dim))*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        self.unembd = unembd
        
        if unembd:
            assert shp[-2] == embd_dim
            if weight_tied:
                #self.linears[-1].weight = self.embedding
                self.embedding = self.linears[-1].weight

    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)
    
    def forward(self, x):
        print(torch.sqrt(torch.mean(x**2)))
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2 or not self.unembd:
                x = f(x)
        x = self.linears[-1](x)
        return x
    
    def pred_logit(self, x):
        return self.forward(x)
    
    
class DistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(DistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist_sq: (B, V)
        n_embd = x.size(-1,)
        w = self.weight
        wx = torch.einsum('bn,vn->bv', x, w) # (B, V)
        ww = torch.norm(w, dim=-1)**2 # (V,)
        xx = torch.norm(x, dim=-1)**2 # (B,)

        dist_sq = ww[None,:] + xx[:,None] - 2 * wx + self.eps
        dist_sq = dist_sq / torch.min(dist_sq, dim=-1, keepdim = True)[0]
        return (dist_sq)**(-self.n)
    
class MLP_HS(nn.Module):
    def __init__(self, shp, vocab_size, embd_dim, input_token=2, init_scale=1., weight_tied=True, n=1., seed=0):
        super(MLP_HS, self).__init__()
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.depth = len(shp) - 1
            
        linear_list = []
        for i in range(self.depth):
            if i < self.depth - 1:
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            else:
                linear_list.append(DistLayer(shp[i], shp[i+1], n=n))
        
        #self.embedding = torch.nn.Parameter(torch.normal(0,1/torch.tensor(embd_dim),size=(vocab_size, embd_dim))*init_scale)
        self.embedding = torch.nn.Parameter(torch.normal(0,1,size=(vocab_size, embd_dim))*init_scale)
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        
        assert shp[-1] == vocab_size
        assert shp[-2] == embd_dim
        assert shp[0] == input_token * embd_dim
        
        self.input_token = input_token
        self.embd_dim = embd_dim
        self.vocab_size = vocab_size
        
        self.weight_tied = weight_tied
        
        if weight_tied:
            self.embedding = self.linears[-1].weight
            
    def id2embd(self, data_id):
        assert data_id.shape[1] == self.input_token
        batch = data_id.shape[0]
        return self.embedding[data_id].reshape(batch,-1)

    def forward(self, x):
        x = self.id2embd(x)
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = self.linears[i](x)
            if i < self.depth - 2:
                x = f(x)
        x = self.linears[-1](x)
        return x
    
    def pred_logit(self, x):
        prob_unnorm = self.forward(x)
        prob = prob_unnorm/torch.sum(prob_unnorm, dim=1, keepdim=True)
        logits = torch.log(prob)
        return logits
    

# 2-Layer Transformer Model with Explicit Residual Connections
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, seq_len = 16, use_dist_layer = False):
        super(ToyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Define transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.use_dist_layer = use_dist_layer
        if use_dist_layer:
            self.dist = DistLayer(d_model, vocab_size, n=1., eps=1e-4, bias=False)
        self.fc = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding

        # Pass through transformer layers with residual connections
        x = embedded
        for layer in self.layers:
            x = layer(x,x) + x  # Explicit residual connection
            
        if self.use_dist_layer:
            x = x[:, -1]
            x = self.dist(x)
            prob = x/torch.sum(x, dim=1, keepdim=True)
            logits = torch.log(prob)
        else:
            logits = self.fc(x[:, -1])  # Only predict the last token
        return logits
    
    def train(self, param_dict: dict):

        num_epochs = param_dict['num_epochs']
        learning_rate = param_dict['learning_rate']
        dataloader = param_dict['dataloader']
        device = param_dict['device']
        

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.type(torch.LongTensor).to(device)
                optimizer.zero_grad()
                logits = self.forward(batch_inputs)

 #               class_counts = torch.bincount(batch_targets.squeeze(), minlength=self.vocab_size).double() + 1e-8
 #               class_weights = 1 / class_counts.cuda()

                criterion = nn.CrossEntropyLoss()#weight=class_weights)
                
                loss = criterion(logits, batch_targets.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
