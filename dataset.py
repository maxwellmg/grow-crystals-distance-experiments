import numpy as np
import torch

def parallelogram_dataset(p, dim, num, seed=0, device='cpu'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N_sample = 5 * num
    x = np.random.choice(p, N_sample*dim*3).reshape(N_sample, 3, dim)
    target = -x[:,0,:] + x[:,1,:] + x[:,2,:]
    id_ = np.where(np.prod((target >= 0) * (target < p), axis=1)==1)[0][:num]
    target = target[id_]
    x = x[id_]
    
    data_id = 0
    for i in range(dim):
        data_id += x[:,:,i] * p ** (dim-i-1)
        
    labels = 0
    for i in range(dim):
        labels += target[:,i] * p ** (dim-i-1)

    data_id = torch.from_numpy(data_id).to(device)
    labels = torch.from_numpy(labels).to(device)
    
    vocab_size = p**dim
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size
    
    return dataset


def modular_addition_dataset(p, seed=0, device='cpu'):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x = np.arange(p)
    y = np.arange(p)
    XX, YY = np.meshgrid(x, y)
    data_id = np.transpose([XX.reshape(-1,), YY.reshape(-1,)])
    labels = (data_id[:,0] + data_id[:,1]) % p
    labels = torch.tensor(labels, dtype=torch.long)
    
    vocab_size = p
    
    dataset = {}
    dataset['data_id'] = data_id
    dataset['label'] = labels
    dataset['vocab_size'] = vocab_size

    return dataset

def split_dataset(dataset, train_ratio, seed=0):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset2 = {}
    
    num = dataset['data_id'].shape[0]
    
    train_num = int(num*train_ratio)
    test_num = num - train_num

    train_id = np.random.choice(num,train_num,replace=False)
    test_id = np.array(list(set(np.arange(num)) - set(train_id)))
    
    dataset2['train_data_id'] = dataset['data_id'][train_id]
    dataset2['test_data_id'] = dataset['data_id'][test_id]
    dataset2['train_label'] = dataset['label'][train_id]
    dataset2['test_label'] = dataset['label'][test_id]
    dataset2['vocab_size'] = dataset['vocab_size']
    return dataset2

def repeat_dataset(dataset):
    
    dataset2 = {}
    
    dataset2['train_data_id'] = dataset['data_id']
    dataset2['test_data_id'] = dataset['data_id']
    dataset2['train_label'] = dataset['label']
    dataset2['test_label'] = dataset['label']
    dataset2['vocab_size'] = dataset['vocab_size']
    
    return dataset2


def combine_dataset(train_dataset, test_dataset):
    
    dataset_c = {}
    
    dataset_c['train_data_id'] = train_dataset['data_id']
    dataset_c['test_data_id'] = test_dataset['data_id']
    dataset_c['train_label'] = train_dataset['label']
    dataset_c['test_label'] = test_dataset['label']
    
    assert train_dataset['vocab_size'] == test_dataset['vocab_size']
    dataset_c['vocab_size'] = train_dataset['vocab_size']
    
    return dataset_c


# Dataset and DataLoader
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]