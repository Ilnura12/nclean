#!/usr/bin/env python
# coding: utf-8

# # imports

# In[1]:


import numpy as np
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import pickle
import statistics

from torch import nn, optim
from torch.nn import Module, Linear, ReLU, MSELoss, Softplus, Sequential, Sigmoid, Tanh, Softmax
from torch.utils.data import DataLoader, Dataset

from CGRtools import RDFRead
from CGRtools.files import RDFwrite

from CIMtools.preprocessing.conditions_container import DictToConditions, ConditionsToDataFrame
from CIMtools.preprocessing import Fragmentor, CGR, EquationTransformer, SolventVectorizer
from CIMtools.model_selection import TransformationOut
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from os import environ
from collections import Counter


# In[2]:


from torch import hstack


# In[3]:


#Зашумление части у
def generating_of_noised_y(y, percent_of_noise):
    y_train_noised = y.copy()
    noised_reactions= []
    frequency_of_noised_points = round((percent_of_noise*len(y))/100)
    for i in random.sample(range(len(y)), frequency_of_noised_points):
        while abs(y[i] - y_train_noised[i]) < 3:
            y_train_noised[i] = random.uniform(min(y), max(y))
        noised_reactions.append(i)
    return y_train_noised, noised_reactions 


# # dataset

# In[4]:


DA = RDFRead ('DA_25.04.2017_All.rdf')


# In[5]:


#Стандартизация
data = []
for reaction in tqdm(DA):
    reaction.standardize()
    reaction.kekule()
    reaction.implicify_hydrogens()
    reaction.thiele()
    data.append(reaction)
del DA


# In[6]:


#Генерация дескрипторов 
def extract_meta(x):
    return [y[0].meta for y in x]

environ["PATH"]+=":/home/ilnura/cim/fragmentor_lin_2017"
features = ColumnTransformer([('temp', EquationTransformer('1/x'), ['temperature']),
                              ('solv', SolventVectorizer(), ['solvent.1'])])
conditions = Pipeline([('meta', FunctionTransformer(extract_meta)),
                       ('cond', DictToConditions(solvents=('additive.1',), 
                                                 temperature='temperature')),
                       ('desc', ConditionsToDataFrame()),
                       ('final', features)])
graph = Pipeline([('CGR', CGR()),
                  ('frg', Fragmentor(fragment_type=3, max_length=4, useformalcharge=True, version='2017'))])
pp = ColumnTransformer([('cond', conditions, [0]), ('graph', graph, [0])])


# In[7]:


from CIMtools.model_selection import TransformationOut

def grouper(cgrs, params):
    groups = []
    for cgr in cgrs:
        group = tuple(cgr.meta[param] for param in params)
        groups.append(group)
    return groups

groups = grouper(data, ['additive.1'])

cv_tr = [y for y in TransformationOut(n_splits=5, n_repeats=1, random_state=1, 
                                      shuffle=True).split(X=data, groups=groups)]
print(cv_tr[0][0].shape, cv_tr[0][1].shape, len(data)) 


# In[8]:


external_test_set = [x for n, x in enumerate(data) if n in cv_tr[0][1]]


# In[9]:


train_test_set = [x for n, x in enumerate(data) if n not in cv_tr[0][1]]


# In[10]:


len(train_test_set), type(train_test_set)


# In[11]:


del groups, cv_tr


# ## X, Y

# In[12]:


#Создание наборов х и у обучающей и тестовой выборок

y_train_test_set = [float(x.meta['logK']) for x in train_test_set]
y_external_test_set = [float(x.meta['logK']) for x in external_test_set]

x_train_test_set = pp.fit_transform([[x] for x in train_test_set])
x_external_test_set = pp.transform([[x] for x in external_test_set])

#del cv_tr, data


# In[41]:


x_train_array, x_valid, y_train, y_valid = train_test_split(x_train_test_set, y_train_test_set, test_size=0.2,
                                                      shuffle=True, random_state=42)


# # NN

# In[14]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[15]:


device


# In[16]:


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[17]:


def prepare_input(X, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    X = torch.from_numpy(X.astype('float32')).cuda()
    y = torch.from_numpy(y.astype('float32')).cuda()
    return X, y


# In[18]:


class MBSplitter(Dataset):
    set_seed(49)

    def __init__(self, X, y):
        super(MBSplitter, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


# In[19]:


class Net(nn.Module):
    def __init__(self, inp_dim_main=None, inp_dim_var=None, hidden_dim=None):
        set_seed(42)
        super().__init__()
        self.net_main = Sequential(Linear(inp_dim_main, hidden_dim),
                                   ReLU(),
                                   Linear(hidden_dim, 1))
        self.net_var = Sequential(Linear(inp_dim_var, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, 1),
                                  Softplus())                               
        self.history = dict.fromkeys(['train_loss', 'valid_loss', 'n_epochs', 'batch_size', 'r2'])       

    def forward(self, X_main, X_var):
        pred = self.net_main(X_main)
        sigma2 = self.net_var(X_var)
        return pred, sigma2


# In[20]:


def loss(pred, sigma2, true):
    loss = (sigma2.log() + (true - pred) ** 2 / sigma2).mean()
    return loss

# def loss1(self, pred, sigma2, true):
#     loss = (sigma2.log()).mean()
#     return loss

# def loss2(self, pred, sigma2, true):
#     loss = ( (true - pred) ** 2 / sigma2).mean()
#     return loss


# In[174]:


test_ds = MBSplitter(x_external_test_set, y_external_test_set)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=True)


# ## С обновлением сигма2 с расчетом готовой моделью

# In[173]:


epochs = 50
# model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3) 
 
percents = [1, 5, 10,20,30,40,50]
#percents = [5, 10, 40]
result_with_upd_with_new_model = dict()
all_results = dict()
for i in tqdm(range (0,10)):
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, y_train_noised)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
        model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) 
        while len(res) < round(len(noised_reactions)/5):

            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x, y_var)   
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                _, sigma2_last = model(x_train_tens.to(device), hstack((x_train_tens.to(device), y_train_noised_tens.to(device))))
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x, y_var_test)
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train) - len(y_train_noised), R2))


            list_to_delete = [x[0].item() for x in (torch.sort(sigma2_last, dim = 0, descending = False)[1])[-5:]]

            x_train = np.delete(x_train, list_to_delete, axis = 0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]

            x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
            train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

        result_with_upd_with_new_model[percent] = res
    all_results[i]  = result_with_upd_with_new_model


# ## С обновлением сигма2 с расчетом во время обучения

# In[135]:


epochs = 50
# model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3) 
 
percents = [1, 5, 10,20,30,40,50]
#percents = [5, 10, 40]
result_with_upd_from_all_batches = dict()
for percent in tqdm(percents):
    res = []
    x_train = x_train_array.copy()
    y_train_noised, noised_reactions = generating_of_noised_y(y_train, percent)
    x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
    train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
    model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    while len(res) < round(len(noised_reactions)/5):
        all_sigmas = torch.empty((0,1))
        all_sigmas = all_sigmas.to(device)
        for epoch in range(epochs+1):
            for batch_ind, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                y_var = hstack((x,y))
                y_var = y_var.to(device)
                pred, sigma2 = model(x, y_var)   
                new_loss = loss(pred, sigma2, y)
                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch == epochs:
                    all_sigmas = torch.cat((all_sigmas, sigma2), dim = 0)
                
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                y_var_test = hstack((x,y))
                y_var_test = y_var_test.to(device)
                pred, sigma2= model(x, y_var_test)
                R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
            res.append((len(y_train) - len(y_train_noised), R2))
            
        
        list_to_delete = [x[0].item() for x in (torch.sort(all_sigmas, dim = 0, descending = False)[1])[-5:]]
        
        x_train = np.delete(x_train, list_to_delete, axis = 0)
        for index in sorted(list_to_delete, reverse=True):
            del y_train_noised[index]
            
        x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
        train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

    result_with_upd_from_all_batches[percent] = res


# ## Без обновления сигма2 с расчетом готовой моделью

# In[ ]:





# In[149]:


epochs = 50
# model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3) 
 
percents = [1, 5, 10,20,30,40,50]
#percents = [10, 30]
result_without_upd_with_new_model= dict()
for percent in tqdm(percents):
    res = []
    x_train = x_train_array.copy()
    y_train_noised, noised_reactions = generating_of_noised_y(y_train, percent)
    x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
    train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
    model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        _, sigma2_last = model(x_train_tens.to(device), hstack((x_train_tens.to(device), y_train_noised_tens.to(device))))
    
    
    while len(res) < round(len(noised_reactions)/5):
     
        for epoch in range(epochs):
            for batch_ind, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                y_var = hstack((x,y))
                y_var = y_var.to(device)
                pred, sigma2 = model(x, y_var)   
                new_loss = loss(pred, sigma2, y)
                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                y_var_test = hstack((x,y))
                y_var_test = y_var_test.to(device)
                pred, sigma2= model(x, y_var_test)
                R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
            res.append((len(y_train) - len(y_train_noised), R2))
            
        confs = [(n, i.item()) for i, n in zip(sigma2_last, range(len(sigma2_last)))]
        idx = sorted(confs, key=lambda x: x[1], reverse=False)    
        list_to_delete =  [n [0] for i,n in enumerate(idx[-5:])] 
        indecies = [x for x in range(sigma2_last.numel()) if x not in list_to_delete]
        sigma2_last = sigma2_last[indecies]
        x_train = np.delete(x_train, list_to_delete, axis = 0)
        for index in sorted(list_to_delete, reverse=True):
            del y_train_noised[index]   
        x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
        train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
        idx = idx[:-5:]
        
    result_without_upd_with_new_model[percent] = res


# ## Без обновления сигма2 с расчетом во время обучения

# In[143]:


epochs = 50
# model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3) 
 
percents = [1, 5, 10,20,30,40,50]
#percents = [10, 30]
result_without_upd_from_all_batches= dict()
for percent in tqdm(percents):
    res = []
    x_train = x_train_array.copy()
    y_train_noised, noised_reactions = generating_of_noised_y(y_train, percent)
    x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
    train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
    model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
    all_sigmas = torch.empty((0,1))
    all_sigmas = all_sigmas.to(device)
    for epoch in range(epochs+1):
        for batch_ind, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            y_var = hstack((x,y))
            y_var = y_var.to(device)
            pred, sigma2 = model(x, y_var)   
            new_loss = loss(pred, sigma2, y)
            new_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch == epochs:
                all_sigmas = torch.cat((all_sigmas, sigma2), dim = 0)

    
    while len(res) < round(len(noised_reactions)/5):
     
        for epoch in range(epochs):
            for batch_ind, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                y_var = hstack((x,y))
                y_var = y_var.to(device)
                pred, sigma2 = model(x, y_var)   
                new_loss = loss(pred, sigma2, y)
                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                y_var_test = hstack((x,y))
                y_var_test = y_var_test.to(device)
                pred, sigma2= model(x, y_var_test)
                R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
            res.append((len(y_train) - len(y_train_noised), R2))
            
        confs = [(n, i.item()) for i, n in zip(sigma2_last, range(len(sigma2_last)))]
        idx = sorted(confs, key=lambda x: x[1], reverse=False)    
        list_to_delete =  [n [0] for i,n in enumerate(idx[-5:])] 
        indecies = [x for x in range(sigma2_last.numel()) if x not in list_to_delete]
        sigma2_last = sigma2_last[indecies]
        x_train = np.delete(x_train, list_to_delete, axis = 0)
        for index in sorted(list_to_delete, reverse=True):
            del y_train_noised[index]   
        x_train_tens, y_train_noised_tens = prepare_input(x_train, np.array(y_train_noised))
        train_ds = MBSplitter(x_train_tens, y_train_noised_tens)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
        idx = idx[:-5:]
        
    result_without_upd_from_all_batches[percent] = res


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




