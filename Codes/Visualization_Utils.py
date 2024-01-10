"""
Created on Fri Jan  5 10:50:03 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the topics on domain adaptation
--> Monitoring Of Laser Powder Bed FusionProcess By Bridging Dissimilar Process MapsUsingDeep Learning-based Domain Adaptation onAcoustic Emissions

@any reuse of this code should be authorized by the first owner, code author
"""

# %% Libraries to import

import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tSNE import *
plt.rcParams.update(plt.rcParamsDefault)

# %%


def predict(dataset, model, model_name):
    X, y = [], []

    model.load_state_dict(torch.load(model_name))  # './just_D1.pth'
    model.eval()
    model = model[0]  # only embedding

    for image, label in dataset:
        x = model(image.cuda())
        X.append(x.detach().cpu().numpy())
        y.append(label.detach().cpu().numpy())

    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]

    return X, y


# %%

def ThreeDplot(folder, filename, S1, L1, check):

    if check == 'D1':
        df2 = pd.DataFrame(L1)
        df2.columns = ['Categorical']
        df2 = df2['Categorical'].replace(2, 3)
        df2 = pd.DataFrame(df2)
        df2 = df2['Categorical'].replace(1, 2)
        df2 = pd.DataFrame(df2)
        df2 = df2['Categorical'].replace(0, 1)
        df2 = pd.DataFrame(df2)
        L1 = df2.to_numpy()
        L1 = np.ravel(L1)
        print(L1)
    else:
        df2 = pd.DataFrame(L1)
        df2.columns = ['Categorical']
        df2 = df2['Categorical'].replace(0, 4)
        df2 = pd.DataFrame(df2)
        df2 = df2['Categorical'].replace(1, 5)
        df2 = pd.DataFrame(df2)
        df2 = df2['Categorical'].replace(2, 6)
        df2 = pd.DataFrame(df2)
        L1 = df2.to_numpy()
        L1 = np.ravel(L1)
        print(L1)

    S1, _, L1, _ = train_test_split(S1, L1, test_size=0.90, random_state=66)

    graph_name1 = str(filename)+'_2D'+'.png'
    graph_name2 = str(filename)+'_3D'+'.png'

    ax, fig = TSNEplot(folder, S1, L1, graph_name1, graph_name2,
                       str(filename), limits=2.5, perplexity=10)

    graph_name = str(filename)+'.gif'

    return ax, fig, graph_name


def ThreeComparisonPlot(folder, filename, S1, L1, S2, L2):

    df2 = pd.DataFrame(L1)
    df2.columns = ['Categorical']
    df2 = df2['Categorical'].replace(2, 3)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, 2)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(0, 1)
    df2 = pd.DataFrame(df2)
    L1 = df2.to_numpy()

    df2 = pd.DataFrame(L2)
    df2.columns = ['Categorical']
    df2 = df2['Categorical'].replace(0, 4)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, 5)
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, 6)
    df2 = pd.DataFrame(df2)
    L2 = df2.to_numpy()

    S1 = np.array(S1)
    L1 = np.ravel(L1)

    S2 = np.array(S2)
    L2 = np.ravel(L2)

    S3 = np.concatenate((S1, S2), axis=0)
    L3 = np.concatenate((L1, L2), axis=0)

    S3, _, L3, _ = train_test_split(S3, L3, test_size=0.90, random_state=66)

    graph_name1 = str(filename)+'_2D'+'.png'
    graph_name2 = str(filename)+'_3D'+'.png'

    ax, fig = TSNEplot_1(folder, S3, L3, graph_name1, graph_name2,
                         str(filename), limits=2.5, perplexity=30)

    graph_name = str(filename)+'.gif'

    return ax, fig, graph_name
