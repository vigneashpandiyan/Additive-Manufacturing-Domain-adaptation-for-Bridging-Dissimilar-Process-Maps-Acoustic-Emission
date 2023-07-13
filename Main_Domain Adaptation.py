# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv


"""

#%% Libraries to import

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import math
import numpy as np
import matplotlib.pyplot as plt
import json

from network import Network,Associative_Regularizer_loss
from Utils import evaluate, write_logs, get_datasets,windowresults
from Visualization_Utils import *

#%%

"""
The purpose of this script is to train the
CNN for domain adaptation with two different distribution
"""


BATCH_SIZE = 500 #batch_size
NUM_EPOCHS = 100 #Total Epoch of training
EMBEDDING_DIM = 832 #Embedding space

DELAY = 500  # Iterative time steps to activate assosiavite losses
GROWTH_STEPS = 1000  # number of steps of linear growth of additional losses
# so domain adaptation losses are in full strength after `DELAY + GROWTH_STEPS` steps

BETA1, BETA2 = 0.5, 0.5 #1,#0.5 #Tunable parameter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting Up CUDA
SAVE_PATH = './CNN_Domain_D1_D2.pth'
LOGS_PATH = 'logs/CNN_Domain_D1_D2.json'


'''
Download data-->https://polybox.ethz.ch/index.php/s/B5YN9pHsIDfJJlG
place data .../datasets/

'''

#%%
    
def train_and_evaluate():
    
    #Datasetloading
    source_loader, val_source_loader, target_loader, val_target_loader = get_datasets(BATCH_SIZE,'D1','D2',test=0.2)
    
    
    num_steps_per_epoch = math.floor(len(source_loader.dataset) / BATCH_SIZE)
    embedder = Network(EMBEDDING_DIM,dropout_rate=0.001).to(DEVICE)
    classifier = nn.Linear(EMBEDDING_DIM, 3).to(DEVICE)
    model = nn.Sequential(embedder, classifier)
    model.train()

    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS - DELAY, eta_min=1e-6)

    cross_entropy = nn.CrossEntropyLoss()
    Domain_Adaptation = Associative_Regularizer_loss()

    text = 'Epoch:{0:2d}, Iteration:{1:3d}, Classification loss: {2:.3f}, ' +\
        'Associative loss: {3:.3f}, Regularizer loss: {4:.4f}, ' +\
        'Total loss: {5:.3f}, learning rate: {6:.6f}'
        
    logs, val_logs = [], []
    
    i = 0  # iteration
    
    Training_loss_mean =[]
    Training_associative_loss_mean =[]
    
    for e in range(NUM_EPOCHS):
        epoch_smoothing=[]
        epoch_associative_loss_smoothing=[]
        model.train()
        for (x_source, x_target), (y_source, _) in zip(source_loader, target_loader):

            x_source = x_source.to(DEVICE)
            
            # print("x_source",x_source.shape)
            
            y_source = y_source.to(DEVICE) #y_source
            
            # print("y_source",y_source.shape)
            
            x_target = x_target.to(DEVICE) #x_target
            
            # print("x_target",x_target.shape) #x_target
            
            batchsize=x_target.shape[0] #x_target
            
            # print("batchsize",batchsize)

            x = torch.cat([x_source, y_source], dim=0) #y_source
            # print("Concatenated",x.shape)
            
            embeddings = embedder(x)
            # print("embeddings",embeddings.shape)
            
            a, b = torch.split(embeddings, batchsize, dim=0) #[batch_size,Embedding_Dimension]
            # print("embeddings_split",a.shape,b.shape)
            
            logits = classifier(a)
            usual_loss = cross_entropy(logits, x_target) #x_target
            
            
            closs = usual_loss.item()
            epoch_smoothing.append(closs)
            
            Associative_loss, Regularizer_loss = Domain_Adaptation(a, b, x_target) #x_target
            
            Total_associative_loss =  (BETA1 * Associative_loss) + (BETA2 * Regularizer_loss)
            Total_associative_loss = Total_associative_loss.item()
            epoch_associative_loss_smoothing.append(Total_associative_loss)
            
            if i > DELAY:
                growth = torch.clamp(torch.tensor((i - DELAY)/GROWTH_STEPS).to(DEVICE), 0.0, 1.0)
                loss = usual_loss + growth * (BETA1 * Associative_loss + BETA2 * Regularizer_loss)

            else:
                loss = usual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i > DELAY:
                scheduler.step()
            lr = scheduler.get_lr()[0]
            
            
            log = (e, i, usual_loss.item(), Associative_loss.item(), Regularizer_loss.item(), loss.item(), lr)
            print(text.format(*log))
            # logs.append(log)
            i += 1
            
        # print(text.format(*log))
        logs.append(log)
        
        Training_loss_mean.append(np.mean(epoch_smoothing)) 
        Training_associative_loss_mean.append(np.mean(epoch_associative_loss_smoothing)) 

        result1 = evaluate(model, cross_entropy, val_source_loader, DEVICE)
        result2 = evaluate(model, cross_entropy, val_target_loader, DEVICE)
        
        print('\n D1 loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('\n D2 loss {0:.3f} and accuracy {1:.3f}\n'.format(*result2))
        
        val_logs.append((e,) + result1 + result2)

    torch.save(model.state_dict(), SAVE_PATH)
    write_logs(logs, val_logs, LOGS_PATH)
    
    classes = ('1', '2', '3')
    windowresults(val_source_loader,model,classes,DEVICE,'CNN_Domain_Adaptation_on_D1_CF.png')
    classes = ('4', '5', '6')
    windowresults(val_target_loader,model,classes,DEVICE,'CNN_Domain_Adaptation_on_D2_CF.png')
    
    return logs, val_logs, Training_loss_mean,Training_associative_loss_mean

#%%
logs, val_logs, Training_loss_mean,Training_associative_loss_mean =train_and_evaluate()
np.save('Domain Adaptation Training_loss_mean.npy',Training_loss_mean)
np.save('Domain Adaptation associative_loss_mean.npy',Training_associative_loss_mean)

#%%

'''
Training plots
'''
with open('logs/CNN_Domain_D1_D2.json', 'r') as f:
    logs = json.load(f)
    
plot_log(logs,Training_loss_mean,Training_associative_loss_mean)
