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
import json
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import os

# %%

'''
Input data_space setting
'''


def data_prep(path, Material):
    """
    Arguments:
        data_file

    Returns:
        Data and ground-truth.
    """
    windowsize = 5000
    featurefile = str(Material)+'_rawspace'+'_' + str(windowsize)+'.npy'
    classfile = str(Material)+'_classspace'+'_' + str(windowsize)+'.npy'

    featurefile = os.path.join(path, featurefile)
    classfile = os.path.join(path, classfile)

    Featurespace = np.load(featurefile).astype(np.float64)
    classspace = np.load(classfile).astype(np.float64)

    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']
    df2 = pd.DataFrame(df2)
    classspace = df2.to_numpy().astype(float)

    return Featurespace, classspace


class Mechanism(Dataset):
    """
    Arguments:
        Dataset

    Returns:
        Data and ground-truth.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        sequence = torch.Tensor(sequence)
        sequence = sequence.view(1, -1)
        label = torch.tensor(label).long()
        label = label.squeeze()
        sequence, label
        return sequence, label


def get_datasets(path, batch_size, test):
    """
    Arguments:
        batch_size
        percentage of split

    Returns:
        Data_loader for training and inference.
    """

    S1, L1 = data_prep(path, "D1")

    D1 = []

    for i in range(len(L1)):
        # print(i)
        sequence_features = S1[i]
        label = L1[i]
        D1.append((sequence_features, label))

    D1 = Mechanism(D1)

    source_loader, val_source_loader = train_test_split(D1, test_size=test, random_state=42)
    source_loader = torch.utils.data.DataLoader(source_loader, batch_size=batch_size, num_workers=0,
                                                shuffle=True, drop_last=True)

    val_source_loader = torch.utils.data.DataLoader(val_source_loader, batch_size=batch_size, num_workers=0,
                                                    shuffle=True, drop_last=True)

    S2, L2 = data_prep(path, "D2")

    D2 = []

    for i in range(len(L2)):
        # print(i)
        sequence_features = S2[i]
        label = L2[i]
        D2.append((sequence_features, label))

    D2 = Mechanism(D2)

    target_loader, val_target_loader = train_test_split(D2, test_size=test, random_state=42)
    target_loader = torch.utils.data.DataLoader(target_loader, batch_size=batch_size, num_workers=0,
                                                shuffle=True)

    val_target_loader = torch.utils.data.DataLoader(val_target_loader, batch_size=batch_size, num_workers=0,
                                                    shuffle=True)

    return source_loader, val_source_loader, target_loader, val_target_loader

# %%


'''
For ploting confusion matrices
'''


def plot_confusion_matrix(path, y_true, y_pred, classes, plotname):

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = cmn*100

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.set(font_scale=3)
    b = sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes, cmap="coolwarm",
                    linewidths=0.1, annot_kws={"size": 25}, cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts:
        b.set_text(b.get_text() + " %")
    plt.ylabel('Actual', fontsize=25)
    plt.xlabel('Predicted', fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center", fontsize=20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname = str(plotname)
    plt.savefig(os.path.join(path, plotname), bbox_inches='tight')
    plt.show()
    plt.close()


def windowresults(path, testset, model, classes, device, filename):

    y_pred = []
    y_true = []

    # iterate over test data
    for batches in testset:

        model.eval()
        data, output = batches
        data, output = data.to(device), output.to(device)
        prediction = model(data)

        prediction = torch.argmax(prediction, dim=1)
        # print("prediction",prediction)
        prediction = prediction.data.cpu().numpy()
        output = output.data.cpu().numpy()
        y_true.extend(output)  # Save Truth
        y_pred.extend(prediction)  # Save Prediction

    plotname = str(filename) + '.png'
    plot_confusion_matrix(path, y_true, y_pred, classes, plotname)

# %%


'''
Helper functions
'''


def write_logs(logs, val_logs, path):
    keys = [
        'step', 'Classification_loss',
        'Associative_loss', 'Regularizer_loss', 'total_loss', 'learning_rate'
    ]
    val_keys = [
        'Epoch', 'D1_logloss', 'D1_accuracy',
        'D2_logloss', 'D2_accuracy'
    ]
    d = {k: [] for k in keys + val_keys}

    for t in logs:
        for i, k in enumerate(keys, 1):
            d[k].append(t[i])

    for t in val_logs:
        for i, k in enumerate(val_keys):
            d[k].append(t[i])

    with open(path, 'w') as f:
        json.dump(d, f)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# %%


''' For visualization'''


def plot_log(folder, logs, Training_loss_mean, Training_associative_loss_mean):

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    plt.plot(logs['Epoch'], logs['D1_logloss'], label='D1 validation loss',
             marker='o', c='red', linewidth=1.5)
    plt.plot(logs['Epoch'], logs['D2_logloss'], label='D2 validation loss',
             marker='*', c='blue', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Validation loss', fontsize=20)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'Validation loss.png'), dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6, 4))
    plt.plot(logs['Epoch'], logs['D1_accuracy'],
             label='D1 Testset', marker='o', c='red', linewidth=1.5)
    plt.plot(logs['Epoch'], logs['D2_accuracy'], label='D2 Testset',
             marker='*',  c='blue', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.title('Accuracy', fontsize=20)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'Accuracy.png'), dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    plt.plot(logs['Epoch'], logs['Classification_loss'], 'blue',
             marker='o', label='Training loss', linewidth=1.5)
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('DA Training loss', fontsize=20)
    plt.legend(loc='upper right', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'Training loss.png'), dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    plt.plot(logs['Epoch'], logs['Associative_loss'],
             label='Associative loss', marker='o', c='green', linewidth=1.5)
    plt.plot(logs['Epoch'], logs['Regularizer_loss'],
             label='Regularizer loss', marker='*', c='purple', linewidth=1.5)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Domain adaptation losses', fontsize=20)
    plt.legend(loc='upper right', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'Domain adaptation losses.png'), dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    plt.plot(logs['Epoch'], logs['learning_rate'], label='learning rate', c='black', linewidth=1.5)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Learning rate', fontsize=20)
    plt.title('Domain adaptation losses', fontsize=20)
    plt.legend(loc='upper right', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'lr Domain adaptation losses.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()
    plt.rc('font', size=15)
    plt.plot(Training_loss_mean, label='Cross-entropy loss', marker='^', c='black',
             linewidth=1.5, markerfacecolor='red', markersize=7, linestyle='dashed')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Cross-entropy loss', fontsize=15)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'Domain Adaptation Training_loss_mean.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()
    plt.rc('font', size=15)
    plt.plot(Training_associative_loss_mean, label='Associative loss', marker='*',
             c='black', linewidth=1.5, markerfacecolor='green', markersize=7)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Associative loss', fontsize=15)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.savefig(os.path.join(folder, 'Domain Adaptation Associative loss_mean.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

# %%
