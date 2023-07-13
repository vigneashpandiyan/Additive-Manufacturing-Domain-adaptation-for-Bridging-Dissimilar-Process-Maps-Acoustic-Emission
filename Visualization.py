# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

"""
#%% Libraries to import

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
from Utils import get_datasets
from network import Network
# from tSNE import TSNEplot
from Visualization_Utils import *

plt.rcParams.update(plt.rcParamsDefault)

#%%
def rotate(angle):
      ax.view_init(azim=angle)

#%%
BATCH_SIZE = 32
EMBEDDING_DIM = 832

source_loader, val_source_loader, target_loader, val_target_loader = get_datasets(BATCH_SIZE,'D1','D2',test=0.6)
embedder = Network(EMBEDDING_DIM,dropout_rate=0.001).cuda()
classifier = nn.Linear(EMBEDDING_DIM, 3).cuda()
model = nn.Sequential(embedder, classifier)

#%%  
S1, L1 = predict(val_source_loader,model,'CNN_Domain_1.pth')
S2, L2  = predict(val_target_loader,model,'CNN_Domain_1.pth')
     
ax,fig,graph_name=ThreeDplot('Domain 1',S1,L1,check='D1')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))


ax,fig,graph_name=ThreeDplot('Domain 2',S2,L2,check='D2')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))


ax,fig,graph_name=ThreeComparisonPlot('CNN_Domain 1 & 2', S1, L1, S2, L2)
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))

#%%

S1, L1 = predict(val_source_loader,model,'CNN_Domain_D1_D2.pth')
S2, L2  = predict(val_target_loader,model,'CNN_Domain_D1_D2.pth')


ax,fig,graph_name=ThreeDplot('Domain 1_Adaptive',S1,L1,check='D1')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))


ax,fig,graph_name=ThreeDplot('Domain 2_Adaptive',S2,L2,check='D2')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))


ax,fig,graph_name=ThreeComparisonPlot('CNN_Domain_Adaptive 1 & 2', S1, L1, S2, L2)
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))