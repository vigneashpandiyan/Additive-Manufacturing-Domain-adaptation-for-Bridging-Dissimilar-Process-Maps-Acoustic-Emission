"""
Created on Fri Jan  5 10:50:03 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the topics on domain adaptation
--> Monitoring Of Laser Powder Bed FusionProcess By Bridging Dissimilar Process MapsUsingDeep Learning-based Domain Adaptation onAcoustic Emissions

@any reuse of this code should be authorized by the first owner, code author
"""

# %% Libraries to import

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utils import get_datasets
from network import Network
from Visualization_Utils import *
plt.rcParams.update(plt.rcParamsDefault)


''' Run the script after CNN_Domain _Adaptation.py.. CNN_Main.py'''

# %%
'''
Download data-->https://polybox.ethz.ch/index.php/s/B5YN9pHsIDfJJlG
--> 10.5281/zenodo.10473583
place data inside .../Data
'''
path = r'C:\Users\srpv\Desktop\C4Science\lpbf-domain-adaptation\Data'  # path of the data


def rotate(angle):
    ax.view_init(azim=angle)


# %%
batch_size = 32
embedding_dim = 832

source_loader, val_source_loader, target_loader, val_target_loader = get_datasets(path,
                                                                                  batch_size, test=0.6)
embedder = Network(embedding_dim, dropout_rate=0.001).cuda()
classifier = nn.Linear(embedding_dim, 3).cuda()
model = nn.Sequential(embedder, classifier)

# %%
folder = r'C:\Users\srpv\Desktop\C4Science\lpbf-domain-adaptation\Codes\CNN-Base'
modelname = '{}/{}'.format(folder, 'CNN_Domain_1.pth')

S1, L1 = predict(val_source_loader, model, modelname)
S2, L2 = predict(val_target_loader, model, modelname)

ax, fig, graph_name = ThreeDplot(folder, 'Domain 1', S1, L1, check='D1')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(os.path.join(folder, graph_name), writer=animation.PillowWriter(fps=20))


ax, fig, graph_name = ThreeDplot(folder, 'Domain 2', S2, L2, check='D2')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(os.path.join(folder, graph_name), writer=animation.PillowWriter(fps=20))


ax, fig, graph_name = ThreeComparisonPlot(folder, 'CNN_Domain 1 & 2', S1, L1, S2, L2)
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(os.path.join(folder, graph_name), writer=animation.PillowWriter(fps=20))

# %%

folder = r'C:\Users\srpv\Desktop\C4Science\lpbf-domain-adaptation\Codes\CNN-Domain_Adaptation'
modelname = '{}/{}'.format(folder, 'CNN_Domain_D1_D2.pth')

S1, L1 = predict(val_source_loader, model, modelname)
S2, L2 = predict(val_target_loader, model, modelname)


ax, fig, graph_name = ThreeDplot(folder, 'Domain 1_Adaptive', S1, L1, check='D1')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(os.path.join(folder, graph_name), writer=animation.PillowWriter(fps=20))


ax, fig, graph_name = ThreeDplot(folder, 'Domain 2_Adaptive', S2, L2, check='D2')
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(os.path.join(folder, graph_name), writer=animation.PillowWriter(fps=20))


ax, fig, graph_name = ThreeComparisonPlot(folder, 'CNN_Domain_Adaptive 1 & 2', S1, L1, S2, L2)
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(os.path.join(folder, graph_name), writer=animation.PillowWriter(fps=20))
