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
import os
from network import Network
from Utils import *
from Trainer import *

# %%

"""
The purpose of this script is to train the
CNN for comparison with two different distribution
"""

batch_size = 500  # batch_size
num_epochs = 10  # Total Epoch of training
embedding_dim = 832  # Embedding space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Setting Up CUDA
data = 'D1'  # 'Choice of distribution'
save_path = './CNN_Domain_1.pth'


'''
Download data-->https://polybox.ethz.ch/index.php/s/B5YN9pHsIDfJJlG
--> 10.5281/zenodo.10473583
place data inside .../Data
'''
path = r'C:\Users\srpv\Desktop\C4Science\lpbf-domain-adaptation\Data'  # path of the data

# %%
'''
create folder for saving the learning curves and dumpping the training logs

'''
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
# Name given for the folder to store the outputs
folder = os.path.join(total_path, 'CNN-Base')
try:
    os.makedirs(folder, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

# %%

# Setting up the CNN network
embedder = Network(embedding_dim, dropout_rate=0.001).to(device)
classifier = nn.Linear(embedding_dim, 3).to(device)

# %%

# Trainer block

Training_loss_mean, model = train_and_evaluate_CNN(
    path, batch_size, data, embedder, classifier, num_epochs, device, folder, save_path)
# %%

count_parameters(model)
np.save(os.path.join(folder, 'CNN_Domain_1_Training_loss_mean.npy'), Training_loss_mean)

# %%

'''
Training plots
'''
plt.rcParams.update(plt.rcParamsDefault)
plt.figure(figsize=(6, 4))
fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.plot(Training_loss_mean, label='Training loss', marker='o', c='black',
         linewidth=1.5, markerfacecolor='blue', markersize=5, linestyle='dashed')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('Training loss', fontsize=15)
plt.legend(loc='best', fontsize=15, frameon=False)
plt.savefig(os.path.join(folder, 'Training loss.png'), dpi=600, bbox_inches='tight')
plt.show()
