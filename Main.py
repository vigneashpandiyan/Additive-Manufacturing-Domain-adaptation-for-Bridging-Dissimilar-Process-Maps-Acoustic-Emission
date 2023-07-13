# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

Adapted from...

#https://github.com/TropComplique/associative-domain-adaptation
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
import numpy as np

from network import Network
from Utils import evaluate, get_datasets, windowresults, count_parameters

#%%

"""
The purpose of this script is to train the
CNN for comparison with two different distribution
"""

BATCH_SIZE = 500 #batch_size
NUM_EPOCHS = 100 #Total Epoch of training
EMBEDDING_DIM = 832 #Embedding space
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting Up CUDA
DATA = 'D1'  # 'Choice of distribution'
SAVE_PATH = './CNN_Domain_1.pth'


'''
Download data-->https://polybox.ethz.ch/index.php/s/B5YN9pHsIDfJJlG
place data .../datasets/

'''

##%%
def train_and_evaluate():
    
    source_loader, val_source_loader, target_loader, val_target_loader = get_datasets(BATCH_SIZE,'D1','D2',test=0.2)
    num_steps_per_epoch = math.floor(len(source_loader.dataset) / BATCH_SIZE)
    
    print('\n Training is on', DATA, '\n')

    embedder = Network(EMBEDDING_DIM,dropout_rate=0.001).to(DEVICE)
    classifier = nn.Linear(EMBEDDING_DIM, 3).to(DEVICE)
    model = nn.Sequential(embedder, classifier)
   
    
    model.train()  #CNN training

    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS, eta_min=1e-6)
    cross_entropy = nn.CrossEntropyLoss()
    
    
    Training_loss_mean =[]
    
    for e in range(NUM_EPOCHS):
        
        
        epoch_smoothing=[]
        
        for x, y in source_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = cross_entropy(logits, y) #cross entropy loss
            
            closs = loss.item()
            epoch_smoothing.append(closs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        
        Training_loss_mean.append(np.mean(epoch_smoothing))
        result1 = evaluate(model, cross_entropy, val_source_loader, DEVICE)
        result2 = evaluate(model, cross_entropy, val_target_loader, DEVICE)
        
        print('\n Epochs', e, '\n')
        print('D1 validation loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('D2 validation loss {0:.3f} and accuracy {1:.3f}\n'.format(*result2))

    torch.save(model.state_dict(), SAVE_PATH)
    classes = ('1', '2', '3')
    windowresults(val_source_loader,model,classes,DEVICE,'CNN_Domain_1 D1_on_D1_CF.png')
    classes = ('4', '5', '6')
    windowresults(val_target_loader,model,classes,DEVICE,'CNN_Domain_1 D2_on_D1_CF.png')

    return Training_loss_mean, model
#%%

#Running the script

Training_loss_mean, model=train_and_evaluate()
#%%

count_parameters(model)
np.save('CNN_Domain_1_Training_loss_mean.npy',Training_loss_mean)


#%%

plt.rcParams.update(plt.rcParamsDefault)
plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.plot(Training_loss_mean, label='Training loss', marker='o', c='black',linewidth =1.5,markerfacecolor='blue', markersize=5,linestyle='dashed')
plt.xlabel('Epochs',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.title('Training loss',fontsize=15);
plt.legend( loc='best',fontsize=15,frameon=False)
plt.savefig('Training loss.png', dpi=600,bbox_inches='tight')
plt.show()



