#!/usr/bin/env python
# coding: utf-8

# # imports

# In[1]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import statistics

from torch import nn, optim, hstack
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


# In[22]:


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

# In[3]:


DA = RDFRead ('DA_25.04.2017_All.rdf')


# In[4]:


#Стандартизация
data = []
for reaction in tqdm(DA):
    reaction.standardize()
    reaction.kekule()
    reaction.implicify_hydrogens()
    reaction.thiele()
    data.append(reaction)
del DA


# In[5]:


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


# In[6]:


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


# In[7]:


external_test_set = [x for n, x in enumerate(data) if n in cv_tr[0][1]]


# In[8]:


train_test_set = [x for n, x in enumerate(data) if n not in cv_tr[0][1]]


# In[9]:


del groups, cv_tr


# ## X, Y

# In[10]:


#Создание наборов х и у обучающей и тестовой выборок

y_train_test_set = [float(x.meta['logK']) for x in train_test_set]
y_external_test_set = [float(x.meta['logK']) for x in external_test_set]

x_train_test_set = pp.fit_transform([[x] for x in train_test_set])
x_external_test_set = pp.transform([[x] for x in external_test_set])

#del cv_tr, data


# In[ ]:


#x_train_array, x_valid, y_train, y_valid = train_test_split(x_train_test_set, y_train_test_set, test_size=0.2,shuffle=True, random_state=42)


# # NN

# In[11]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[12]:


device


# In[13]:


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[14]:


def prepare_input(X, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    X = torch.from_numpy(X.astype('float32')).cuda()
    y = torch.from_numpy(y.astype('float32')).cuda()
    return X, y


# In[15]:


class MBSplitter(Dataset):
    set_seed(49)

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


# In[16]:


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


# In[17]:


def loss(pred, sigma2, true):
    loss = (sigma2.log() + (true - pred) ** 2 / sigma2).mean()
    return loss

# def loss1(self, pred, sigma2, true):
#     loss = (sigma2.log()).mean()
#     return loss

# def loss2(self, pred, sigma2, true):
#     loss = ( (true - pred) ** 2 / sigma2).mean()
#     return loss


# In[18]:


test_ds = MBSplitter(x_external_test_set, np.array(y_external_test_set).reshape(-1,1))
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)


# ## С обновлением сигма2 с расчетом готовой моделью

# In[211]:


epochs = 50
percents = [1, 5, 10,20,30,40,50]

inp_dim_main, inp_dim_var = x_train_test_set.shape[1], x_train_test_set.shape[1]+1


# In[ ]:


# model = Net(inp_dim_main=train_ds[0][0].shape[0], inp_dim_var=train_ds[0][0].shape[0]+1, hidden_dim=512)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)  


# In[ ]:


all_results1000_from_all_batches = dict()
all_results2000_from_all_batches = dict()


# In[259]:



all_results1000_new_model = dict()
all_results2000_new_model = dict()


# In[260]:


for i in tqdm(range (0,10)):
    result_with_upd_with_new_model = dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        while len(res) < round(len(noised_reactions)/5):
            all_sigmas = torch.empty((0,1))
            all_sigmas = all_sigmas.to(device)
            all_preds = torch.empty((0,1))
            all_preds = all_preds.to(device)
            all_y = torch.empty((0,1))
            all_y = all_y.to(device)
            model = Net(x_train.shape[1], x_train.shape[1]+1, hidden_dim=2000)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3) 
            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float())   
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                train_dl_new = DataLoader(train_ds, batch_size=256, shuffle=False)
                for x, y in train_dl_new:
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    _, sigma2_last = model(x.float(), y_var.float())
                    
                    all_sigmas = torch.cat((all_sigmas, sigma2_last),dim =0)
                
                    
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    all_preds = torch.cat((all_preds, pred),dim =0)
                    all_y = torch.cat((all_y, y), dim = 0)
                R2 = r2_score(all_preds.cpu().numpy(), all_y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))
                #print(res)

            list_to_delete = [x[0].item() for x in (torch.sort(all_sigmas, dim=0, descending = True)[1])[:5]]
            #print(list_to_delete)
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

        result_with_upd_with_new_model[percent] = res
    all_results2000_new_model[i]  = result_with_upd_with_new_model


# ## Без обновления сигма2 с расчетом готовой моделью

# In[352]:


for i in tqdm(range(0,1)):
    result_without_upd_with_new_model= dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        all_sigmas = torch.empty((0,1))
        all_sigmas = all_sigmas.to(device)
        while len(res) < round(len(noised_reactions)/5):
            model = Net(inp_dim_main, inp_dim_var, hidden_dim=2000)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3) 
            
            all_preds = torch.empty((0,1))
            all_preds = all_preds.to(device)
            all_y = torch.empty((0,1))
            all_y = all_y.to(device)
            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float()) 
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                if len(res) == 0:
                    train_dl_new = DataLoader(train_ds, batch_size=256, shuffle=False)
                    for x, y in train_dl_new:
                        x, y = x.to(device), y.to(device)
                        y_var = hstack((x,y))
                        y_var = y_var.to(device)
                        _, sigma2_last = model(x.float(), y_var.float())
                        all_sigmas = torch.cat((all_sigmas, sigma2_last),dim =0)
#                 else:
#                     continue
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float()) 
                    all_preds = torch.cat((all_preds, pred),dim =0)
                    all_y = torch.cat((all_y, y), dim = 0)
                R2 = r2_score(all_preds.cpu().numpy(), all_y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))
                print(res)
            confs = [(n, i.item()) for i, n in zip(all_sigmas, range(len(all_sigmas)))]
            idx = sorted(confs, key=lambda x: x[1], reverse=False)    
            list_to_delete =  [n [0] for i,n in enumerate(idx[-5:])] 
            indecies = [x for x in range(all_sigmas.numel()) if x not in list_to_delete]
            all_sigmas = all_sigmas[indecies]
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]   
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
            idx = idx[:-5:]

        result_without_upd_with_new_model[percent] = res
    without_upd_results2000_with_new_model[i] = result_without_upd_with_new_model


# In[338]:


base_line = res[0][1] 


# ## Рандомное удаление

# In[268]:


random_results = dict()
for i in tqdm(range(0,10)):
    result_rand= dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        while len(res) < round(len(noised_reactions)/5):
            model = Net(inp_dim_main, inp_dim_var, hidden_dim=2000)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3) 
           
            all_preds = torch.empty((0,1))
            all_preds = all_preds.to(device)
            all_y = torch.empty((0,1))
            all_y = all_y.to(device)
            
            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float()) 
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float()) 
                    all_preds = torch.cat((all_preds, pred),dim =0)
                    all_y = torch.cat((all_y, y), dim = 0)
                R2 = r2_score(all_preds.cpu().numpy(), all_y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))

            list_to_delete = np.random.choice(np.arange(0,len(y_train_noised)), 5, replace=False) 
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]   
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
            idx = idx[:-5:]

        result_rand[percent] = res
    random_results[i] = result_rand


# In[ ]:


for i in tqdm(range (0,1)):
    result_with_upd_with_new_model = dict()
    for percent in [0]:
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=1000)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) 
        while len(res) < round(len(noised_reactions)/5):
            all_sigmas = torch.empty((0,1))
            all_sigmas = all_sigmas.to(device)

            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float())   
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                train_dl_new = DataLoader(train_ds, batch_size=256, shuffle=False)
                for x, y in train_dl_new:
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    _, sigma2_last = model(x.float(), y_var.float())
                    
                    all_sigmas = torch.cat((all_sigmas, sigma2_last),dim =0)
           
                print(len(all_sigmas))
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))


            list_to_delete = [x[0].item() for x in (torch.sort(all_sigmas, dim=0, descending = True)[1])[:5]]
            print(list_to_delete)
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

        


# In[213]:


for i in tqdm(range (0,10)):
    result_with_upd_with_new_model = dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=2000)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) 
        while len(res) < round(len(noised_reactions)/5):
            all_sigmas = torch.empty((0,1))
            all_sigmas = all_sigmas.to(device)

            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float())   
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                train_dl_new = DataLoader(train_ds, batch_size=256, shuffle=False)
                for x, y in train_dl_new:
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    _, sigma2_last = model(x.float(), y_var.float())
                    all_sigmas = torch.cat((all_sigmas, sigma2_last),dim =0)
           
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))


            list_to_delete = [x[0].item() for x in (torch.sort(all_sigmas, dim=0, descending = True)[1])[:5]]
            print(len(list_to_delete))
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

        result_with_upd_with_new_model[percent] = res
    all_results2000_new_model[i]  = result_with_upd_with_new_model


# In[175]:


[all_results1000_new_model, all_results2000_new_model, all_results1000_from_all_batches, 
         all_results2000_from_all_batches, without_upd_results2000_with_new_model,
         without_upd_results1000_with_new_model, without_upd_from_all_batches2000, without_upd_from_all_batches1000]:


# ## С обновлением сигма2 с расчетом во время обучения

# In[24]:


for i in tqdm(range (0, 10)):
    result_with_upd_from_all_batches = dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=1000)
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
                    pred, sigma2 = model(x.float(), y_var.float())   
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if epoch == epochs:
                        all_sigmas = torch.cat((all_sigmas, sigma2),dim =0)

            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))


            list_to_delete = [x[0].item() for x in (torch.sort(all_sigmas, dim=0, descending = False)[1])[-5:]]
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

        result_with_upd_from_all_batches[percent] = res
    all_results1000_from_all_batches[i]  = result_with_upd_from_all_batches


# In[26]:


for i in tqdm(range (0, 10)):
    result_with_upd_from_all_batches = dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=2000)
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
                    pred, sigma2 = model(x.float(), y_var.float())   
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if epoch == epochs:
                        all_sigmas = torch.cat((all_sigmas, sigma2),dim =0)

            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))


            list_to_delete = [x[0].item() for x in (torch.sort(all_sigmas, dim=0, descending = False)[1])[-5:]]
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)

        result_with_upd_from_all_batches[percent] = res
    all_results2000_from_all_batches[i]  = result_with_upd_from_all_batches


# In[220]:


without_upd_results1000_with_new_model = dict()
for i in tqdm(range(0,10)):
    result_without_upd_with_new_model= dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=1000)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) 
        all_sigmas = torch.empty((0,1))
        all_sigmas = all_sigmas.to(device)
        all_preds = torch.empty((0,1))
        all_preds = all_preds.to(device)
        all_y = torch.empty((0,1))
        all_y = all_y.to(device)
        while len(res) < round(len(noised_reactions)/5):
            
            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float()) 
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                if len(res) == 0:
                    train_dl_new = DataLoader(train_ds, batch_size=256, shuffle=False)
                    for x, y in train_dl_new:
                        x, y = x.to(device), y.to(device)
                        y_var = hstack((x,y))
                        y_var = y_var.to(device)
                        _, sigma2_last = model(x.float(), y_var.float())
                        all_sigmas = torch.cat((all_sigmas, sigma2_last),dim =0)
#                 else:
#                     continue
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float()) 
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))

        
            list_to_delete =  [n [0] for i,n in enumerate(idx[-5:])] 
            indecies = [x for x in range(all_sigmas.numel()) if x not in list_to_delete]
            all_sigmas = all_sigmas[indecies]
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]   
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
            idx = idx[:-5:]

        result_without_upd_with_new_model[percent] = res
    without_upd_results1000_with_new_model[i] = result_without_upd_with_new_model


# ## Без обновления сигма2 с расчетом во время обучения

# In[125]:


without_upd_from_all_batches1000= dict()
for i in tqdm(range(0,10)):
    result_without_upd_from_all_batches = dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=1000)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) 

        all_sigmas = torch.empty((0,1))
        all_sigmas = all_sigmas.to(device)
        for epoch in range(epochs+1):
            for batch_ind, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                y_var = hstack((x,y))
                y_var = y_var.to(device)
                pred, sigma2 = model(x.float(), y_var.float())   
                new_loss = loss(pred, sigma2, y)
                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch == epochs:
                    all_sigmas = torch.cat((all_sigmas, sigma2), dim=0)


        while len(res) < round(len(noised_reactions)/5):

            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float()) 
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))

            confs = [(n, i.item()) for i, n in zip(all_sigmas, range(len(all_sigmas)))]
            idx = sorted(confs, key=lambda x: x[1], reverse=False)    
            list_to_delete =  [n [0] for i,n in enumerate(idx[-5:])] 
            indecies = [x for x in range(all_sigmas.numel()) if x not in list_to_delete]
            all_sigmas = all_sigmas[indecies]
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]   
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
            idx = idx[:-5:]

        result_without_upd_from_all_batches[percent] = res
    without_upd_from_all_batches1000[i] = result_without_upd_from_all_batches


# In[127]:


without_upd_from_all_batches2000= dict()
for i in tqdm(range(0,10)):
    result_without_upd_from_all_batches = dict()
    for percent in tqdm(percents):
        res = []
        x_train = x_train_test_set.copy()
        y_train_noised, noised_reactions = generating_of_noised_y(y_train_test_set, percent)
        train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = Net(inp_dim_main, inp_dim_var, hidden_dim=2000)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) 

        all_sigmas = torch.empty((0,1))
        all_sigmas = all_sigmas.to(device)
        for epoch in range(epochs+1):
            for batch_ind, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                y_var = hstack((x,y))
                y_var = y_var.to(device)
                pred, sigma2 = model(x.float(), y_var.float())   
                new_loss = loss(pred, sigma2, y)
                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch == epochs:
                    all_sigmas = torch.cat((all_sigmas, sigma2), dim=0)


        while len(res) < round(len(noised_reactions)/5):

            for epoch in range(epochs):
                for batch_ind, (x, y) in enumerate(train_dl):
                    x, y = x.to(device), y.to(device)
                    y_var = hstack((x,y))
                    y_var = y_var.to(device)
                    pred, sigma2 = model(x.float(), y_var.float()) 
                    new_loss = loss(pred, sigma2, y)
                    new_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    y_var_test = hstack((x,y))
                    y_var_test = y_var_test.to(device)
                    pred, sigma2= model(x.float(), y_var_test.float())
                    R2 = r2_score(pred.cpu().numpy(), y.cpu().numpy())
                res.append((len(y_train_test_set) - len(y_train_noised), R2))

            confs = [(n, i.item()) for i, n in zip(all_sigmas, range(len(all_sigmas)))]
            idx = sorted(confs, key=lambda x: x[1], reverse=False)    
            list_to_delete =  [n [0] for i,n in enumerate(idx[-5:])] 
            indecies = [x for x in range(all_sigmas.numel()) if x not in list_to_delete]
            all_sigmas = all_sigmas[indecies]
            x_train = np.delete(x_train, list_to_delete, axis=0)
            for index in sorted(list_to_delete, reverse=True):
                del y_train_noised[index]   
            train_ds = MBSplitter(x_train, np.array(y_train_noised).reshape(-1,1))
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=False)
            idx = idx[:-5:]

        result_without_upd_from_all_batches[percent] = res
    without_upd_from_all_batches2000[i] = result_without_upd_from_all_batches


# # График
# 
# 
# 

# In[368]:


perc = 5
#def grapr(perc):
result = pd.DataFrame()
for res in [all_results2000_new_model, without_upd_results2000_with_new_model]:
    df = pd.DataFrame(res[0][perc])
    df.columns = 'number of points', '0'
    for i in range (1,10):
        df_new = pd.DataFrame(res[i][perc])
        df_new.columns = 'number of points', str(i)
        df = pd.merge(df, df_new, on='number of points')
    col = df.loc[: , '0':'9']
    df['mean'] = col.mean(axis=1)
    result = pd.concat([result, df['mean']], axis = 1)

df_random = pd.DataFrame(random_results[0][perc])
df_random.columns = 'number of points', '0'
for i in range (1,3):
    df_new_random = pd.DataFrame(random_results[i][perc])
    df_new_random.columns = 'number of points', str(i)
    df_random = pd.merge(df_random, df_new_random, on='number of points')
col_random = df_random.loc[: , '0':'2']
df_random['mean'] = col_random.mean(axis=1)
result_random = pd.concat([df_random, df_random['mean']], axis = 1)    
result = pd.concat([df['number of points'], result, df_random['mean']], axis = 1)
result.iat[0, 3] = result.iat[0, 1]
result.iat[0, 2] = result.iat[0, 1]
result.columns = ['число удаленных точек','с пересчетом','без пересчета', 'рандомное удаление']
#                     'без пересчета готовой моделью, число нейронов = 1000','без пересчета сигма 2 со всех батчей, число нейронов = 2000',
#                     'без пересчета сигма 2 со всех батчей, число нейронов = 1000']
plt = result.plot(x = 'число удаленных точек', title = str(perc)+'%', grid=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.set_ylabel('R^2')
plt.axhline(base_line, color = 'black')
plt.set_ylim(None, 1)

