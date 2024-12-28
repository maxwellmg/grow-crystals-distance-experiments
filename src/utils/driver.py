import sys
sys.path.append("..")
from src.utils.dataset import *
from src.utils.model import *
import os

def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_single_model(param_dict: dict):

    if "seed" not in param_dict:
        raise ValueError("seed must be provided in param_dict")
    if "data_id" not in param_dict:
        raise ValueError("data_id must be provided in param_dict")
    if "data_size" not in param_dict:
        raise ValueError("data_size must be provided in param_dict")
    if "train_ratio" not in param_dict:
        raise ValueError("train_ratio must be provided in param_dict")
    if "model_id" not in param_dict:
        raise ValueError("model_id must be provided in param_dict")
    if "device" not in param_dict:
        raise ValueError("device must be provided in param_dict")
    if "embd_dim" not in param_dict:
        raise ValueError("embd_dim must be provided in param_dict")
    
    seed = param_dict['seed']
    data_id = param_dict['data_id']
    data_size = param_dict['data_size']
    train_ratio = param_dict['train_ratio']
    model_id = param_dict['model_id']
    device = param_dict['device']
    embd_dim = param_dict['embd_dim']

    set_seed(seed)

    
    # define dataset
    input_token = 2
    if data_id == "lattice":
        dataset = parallelogram_dataset(p=5, dim=2, num=data_size, seed=seed, device=device)
        input_token = 3
    elif data_id == "greater":
        dataset = greater_than_dataset(p=30, num=data_size, seed=seed, device=device)
    elif data_id == "family_tree":
        dataset = family_tree_dataset_2(p=127, num=data_size, seed=seed, device=device)
    elif data_id == "equivalence":
        input_token = 1
        dataset = mod_classification_dataset(p=100, num=data_size, seed=seed, device=device)
    elif data_id == "circle":
        dataset = modular_addition_dataset(p=31, num=data_size, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown data_id: {data_id}")
    
    dataset = split_dataset(dataset, train_ratio=train_ratio, seed=seed)
    vocab_size = dataset['vocab_size']


    # define model
    if model_id == "H_MLP":
        weight_tied = True
        hidden_size = 100
        shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
        model = MLP_HS(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, weight_tied=weight_tied, seed=seed).to(device)
    elif model_id == "standard_MLP":
        unembd = True
        weight_tied = True
        hidden_size = 100
        shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
        model = MLP(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, unembd=unembd, weight_tied=weight_tied, seed=seed).to(device)
    elif model_id == "H_transformer":
        model = ToyTransformer(vocab_size=vocab_size, d_model=embd_dim, nhead=8, num_layers=1, seq_len=input_token, use_dist_layer=True).to(device)
    elif model_id == "standard_transformer":
        model = ToyTransformer(vocab_size=vocab_size, d_model=embd_dim, nhead=8, num_layers=1, seq_len=input_token, use_dist_layer=False).to(device)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    
    # define dataloader
    batch_size = 16
    train_dataset = ToyDataset(dataset['train_data_id'], dataset['train_label'])
    test_dataset = ToyDataset(dataset['test_data_id'], dataset['test_label'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ret_dic = {}
    ret_dic["results"] = model.train(param_dict={'num_epochs': 4000, 'learning_rate': 0.001, 'train_dataloader': train_dataloader, 'test_dataloader': test_dataloader, 'device': device})
    ret_dic["model"] = model
    ret_dic["dataset"] = dataset

    return ret_dic