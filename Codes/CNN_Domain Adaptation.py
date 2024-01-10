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
import json
import os
from network import Network, Associative_Regularizer_loss
from Trainer import *
from Utils import *

# %%

"""
The purpose of this script is to train the
CNN for domain adaptation with  data shifting in distribution
"""


batch_size = 500  # batch_size
num_epochs = 10  # Total Epoch of training
embedding_dim = 832  # Embedding space
delay = 500  # Iterative time steps to activate assosiavite losses
growth_steps = 1000  # number of steps of linear growth of additional losses
# so domain adaptation losses are in full strength after `DELAY + GROWTH_STEPS` steps

Beta1, Beta2 = 0.5, 0.5  # 1,#0.5 #Tunable parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Setting Up CUDA
save_path = './CNN_Domain_D1_D2.pth'

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
folder = os.path.join(total_path, 'CNN-Domain_Adaptation')
try:
    os.makedirs(folder, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")


Log_Path = folder + '/CNN_Domain_D1_D2.json'  # Name of the log file


# %%

# Setting up the CNN network
embedder = Network(embedding_dim, dropout_rate=0.001).to(device)
classifier = nn.Linear(embedding_dim, 3).to(device)


# %%

# Trainer block

logs, val_logs, Training_loss_mean, Training_associative_loss_mean = train_and_evaluate(path, batch_size,
                                                                                        embedder, classifier,
                                                                                        num_epochs, device,
                                                                                        Beta1, Beta2,
                                                                                        delay, growth_steps,
                                                                                        Log_Path, folder, save_path)

np.save(os.path.join(folder, 'Domain Adaptation Training_loss_mean.npy'), Training_loss_mean)
np.save(os.path.join(folder, 'Domain Adaptation associative_loss_mean.npy'),
        Training_associative_loss_mean)

# %%

'''
Training plots
'''
with open('CNN-Domain_Adaptation/CNN_Domain_D1_D2.json', 'r') as f:
    logs = json.load(f)

plot_log(folder, logs, Training_loss_mean, Training_associative_loss_mean)
