import time
import os
import sys

sys.path.append('..')

from src.utils.model import *
from src.utils.dataset import *
import numpy as np
from sklearn.decomposition import PCA
import math

def run():
    
    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    
    model_modes = ["standard", "ip", "hs1", "hs2", "hs3"]
    #model_modes = ["hs2"]
    embd_dims = [1,2,3,4,5,10,20,50]
    lambs = [0.,0.1,0.3,1.]
    seeds = [0]

    xx, yy, zz, uu= np.meshgrid(model_modes, embd_dims, lambs, seeds)
    params_ = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,), zz.reshape(-1,), uu.reshape(-1,)]))
    
    indices = np.arange(params_.shape[0])
    
    my_indices = indices[my_task_id:indices.shape[0]:num_tasks]

    for i in my_indices:
        
        steps = 10001
        
        model_mode = params_[i][0].astype('str') 
        print('model_mode:', model_mode)
        embd_dim = params_[i][1].astype('int')
        lamb = params_[i][2].astype('float') 
        seed = params_[i][3].astype('int') 
        
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.set_default_tensor_type(torch.DoubleTensor)

        device = 'cpu'

        p = 59
        #embd_dim = 10
        input_token = 2
        vocab_size = p


        if model_mode == 'ip':
            # ip model
            unembd = True
            weight_tied = True
            hidden_size = 100
            shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
            model = MLP(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, unembd=unembd, weight_tied=weight_tied, seed=seed).to(device)
        elif model_mode == 'hs3':
            weight_tied = True
            hidden_size = 100
            shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
            model = MLP_HS(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, weight_tied=weight_tied, seed=seed, n=1).to(device)
        elif model_mode == 'hs2':
            weight_tied = True
            hidden_size = 100
            shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
            model = MLP_HS(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, weight_tied=weight_tied, seed=seed, n=math.sqrt(embd_dim)).to(device)
        elif model_mode == 'hs1':
            weight_tied = True
            hidden_size = 100
            shp = [input_token * embd_dim, hidden_size, embd_dim, vocab_size]
            model = MLP_HS(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, weight_tied=weight_tied, seed=seed, n=embd_dim).to(device)
        elif model_mode == 'standard':
            unembd = False
            weight_tied = False
            hidden_size = 100
            shp = [input_token * embd_dim, hidden_size, vocab_size]
            model = MLP(shp=shp, vocab_size=vocab_size, embd_dim=embd_dim, input_token=input_token, unembd=unembd, weight_tied=weight_tied, seed=seed).to(device)
        else:
            print('model_mode not recognized!')


        # data
        dataset = modular_addition_dataset(p=p)
        #dataset = repeat_dataset(dataset)
        dataset = split_dataset(dataset, train_ratio=0.8, seed=seed)

        ### train ###
        wd = 0.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        steps = 4001 #4001
        log = 200

        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        embds = []

        #lamb = 0.1

        for step in range(steps):

            optimizer.zero_grad()

            logits = model.pred_logit(dataset['train_data_id'])
            loss = torch.nn.functional.cross_entropy(logits, dataset['train_label'])

            embd_reg = torch.mean(torch.sqrt(torch.mean(model.embedding**2, dim=0)))
            total_loss = loss + lamb * embd_reg

            acc = torch.mean((torch.argmax(logits, dim=1) == dataset['train_label']).float())

            train_losses.append(loss.item())
            train_accs.append(acc.item())

            logits_test = model.pred_logit(dataset['test_data_id'])
            loss_test = torch.nn.functional.cross_entropy(logits_test, dataset['test_label'])

            acc_test = torch.mean((torch.argmax(logits_test, dim=1) == dataset['test_label']).float())

            test_losses.append(loss_test.item())
            test_accs.append(acc_test.item())

            #total_loss = loss
            total_loss.backward()
            optimizer.step()

            if step % log == 0:
                print("step = %d | total loss: %.2e | train loss: %.2e | test loss %.2e | train acc: %.2e | test acc: %.2e "%(step, total_loss.cpu().detach().numpy(), loss.cpu().detach().numpy(), loss_test.cpu().detach().numpy(), acc.cpu().detach().numpy(), acc_test.cpu().detach().numpy()))

        if step % 100 == 0:
            embds.append(model.embedding.cpu().detach().numpy())

        embd = model.embedding.cpu().detach().numpy()
        X = embd
        pca = PCA(n_components=embd_dim)
        pca.fit(X)
        embd_t = pca.fit_transform(X)

        active_pca_dim = np.sum(pca.explained_variance_ratio_ > 1e-4)
        active_embd_dim = torch.sum(torch.mean(model.embedding**2, dim=0) > 1e-4).item()

        # save train_acc, test_acc, parallelogram_acc, active_pca_dim, active_embd_dim, 
        np.savetxt('./results/modadd_largeinit/model_%s_embddim_%d_lamb_%.2f_seed_%d_p_59_performance.txt'%(model_mode, embd_dim, lamb, seed), [train_accs[-1], test_accs[-1], active_pca_dim, active_embd_dim])
        # save embd
        np.savetxt('./results/modadd_largeinit/model_%s_embddim_%d_lamb_%.2f_seed_%d_p_59_embedding.txt'%(model_mode, embd_dim, lamb, seed), embd)

run()
