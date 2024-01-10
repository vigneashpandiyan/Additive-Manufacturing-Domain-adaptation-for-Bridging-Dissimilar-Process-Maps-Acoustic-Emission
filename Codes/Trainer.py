"""
Created on Fri Jan  5 10:50:03 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the topics on domain adaptation
--> Monitoring Of Laser Powder Bed FusionProcess By Bridging Dissimilar Process MapsUsingDeep Learning-based Domain Adaptation onAcoustic Emissions

@any reuse of this code should be authorized by the first owner, code author
"""

# %% Libraries to import
import math
import torch.nn as nn
from Utils import write_logs, get_datasets, windowresults
from network import Network, Associative_Regularizer_loss
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

''' Trainer block for CNN domain adaptation'''


def train_and_evaluate(path, batch_size, embedder, classifier,
                       num_epochs, device, Beta1, Beta2,
                       delay, growth_steps, Log_Path, folder, save_path):

    # Datasetloading
    source_loader, val_source_loader, target_loader, val_target_loader = get_datasets(
        path, batch_size, test=0.2)

    num_steps_per_epoch = math.floor(len(source_loader.dataset) / batch_size)

    # Network initialization
    model = nn.Sequential(embedder, classifier)
    model.train()

    # optimizer setup
    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch *
                                  num_epochs - delay, eta_min=1e-6)

    # Loss functions
    cross_entropy = nn.CrossEntropyLoss()
    Domain_Adaptation = Associative_Regularizer_loss()

    text = 'Epoch:{0:2d}, Iteration:{1:3d}, Classification loss: {2:.3f}, ' +\
        'Associative loss: {3:.3f}, Regularizer loss: {4:.4f}, ' +\
        'Total loss: {5:.3f}, learning rate: {6:.6f}'

    logs, val_logs = [], []

    i = 0  # iteration

    Training_loss_mean = []
    Training_associative_loss_mean = []

    for e in range(num_epochs):
        epoch_smoothing = []
        epoch_associative_loss_smoothing = []
        model.train()
        for (x_source, y_source), (x_target, _) in zip(source_loader, target_loader):

            x_source = x_source.to(device)
            x_target = x_target.to(device)
            y_source = y_source.to(device)
            batchsize = y_source.shape[0]

            x = torch.cat([x_source, x_target], dim=0)
            embeddings = embedder(x)

            a, b = torch.split(embeddings, batchsize, dim=0)  # [batch_size,Embedding_Dimension]

            logits = classifier(a)
            usual_loss = cross_entropy(logits, y_source)

            closs = usual_loss.item()
            epoch_smoothing.append(closs)

            Associative_loss, Regularizer_loss = Domain_Adaptation(a, b, y_source)

            Total_associative_loss = (Beta1 * Associative_loss) + (Beta2 * Regularizer_loss)
            Total_associative_loss = Total_associative_loss.item()
            epoch_associative_loss_smoothing.append(Total_associative_loss)

            if i > delay:
                growth = torch.clamp(torch.tensor((i - delay)/growth_steps).to(device), 0.0, 1.0)
                loss = usual_loss + growth * (Beta1 * Associative_loss + Beta2 * Regularizer_loss)

            else:
                loss = usual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i > delay:
                scheduler.step()
            lr = scheduler.get_lr()[0]

            log = (e, i, usual_loss.item(), Associative_loss.item(),
                   Regularizer_loss.item(), loss.item(), lr)
            print(text.format(*log))
            i += 1

        logs.append(log)

        Training_loss_mean.append(np.mean(epoch_smoothing))
        Training_associative_loss_mean.append(np.mean(epoch_associative_loss_smoothing))

        result1 = evaluate(model, cross_entropy, val_source_loader, device)
        result2 = evaluate(model, cross_entropy, val_target_loader, device)

        print('\n D1 loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('\n D2 loss {0:.3f} and accuracy {1:.3f}\n'.format(*result2))

        val_logs.append((e,) + result1 + result2)

    write_logs(logs, val_logs, Log_Path)

    torch.save(model.state_dict(), '{}/{}'.format(folder, save_path))
    classes = ('1', '2', '3')
    windowresults(folder, val_source_loader, model, classes,
                  device, 'CNN_Domain_Adaptation_D1_on_D1_CF')
    classes = ('4', '5', '6')
    windowresults(folder, val_target_loader, model, classes,
                  device, 'CNN_Domain_Adaptation_D2_on_D1_CF')

    return logs, val_logs, Training_loss_mean, Training_associative_loss_mean

# %%


def train_and_evaluate_CNN(path, batch_size, data, embedder, classifier, num_epochs, device, folder, save_path):

    source_loader, val_source_loader, target_loader, val_target_loader = get_datasets(
        path, batch_size, test=0.2)
    num_steps_per_epoch = math.floor(len(source_loader.dataset) / batch_size)

    print('\n Training is on', data, '\n')

    model = nn.Sequential(embedder, classifier)

    model.train()  # CNN training

    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * num_epochs, eta_min=1e-6)
    cross_entropy = nn.CrossEntropyLoss()

    Training_loss_mean = []

    for e in range(num_epochs):

        epoch_smoothing = []

        for x, y in source_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = cross_entropy(logits, y)  # cross entropy loss

            closs = loss.item()
            epoch_smoothing.append(closs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Training_loss_mean.append(np.mean(epoch_smoothing))
        result1 = evaluate(model, cross_entropy, val_source_loader, device)
        result2 = evaluate(model, cross_entropy, val_target_loader, device)

        print('\n Epochs', e, '\n')
        print('D1 validation loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('D2 validation loss {0:.3f} and accuracy {1:.3f}\n'.format(*result2))

    torch.save(model.state_dict(), '{}/{}'.format(folder, save_path))
    classes = ('1', '2', '3')
    windowresults(folder, val_source_loader, model, classes, device, 'CNN_Domain_1 D1_on_D1_CF')
    classes = ('4', '5', '6')
    windowresults(folder, val_target_loader, model, classes, device, 'CNN_Domain_1 D2_on_D1_CF')

    return Training_loss_mean, model
# %%


'''
Model Evaluation
'''


def evaluate(model, criterion, loader, device):
    """
    Arguments:
        model
        loss function
        loader
        device

    Returns:
        loss and accuracy.
    """

    model.eval()
    total_loss = 0.0
    num_hits = 0
    num_samples = 0

    for images, targets in loader:

        batch_size = images.size(0)
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            logits = model(images)
            loss = criterion(logits, targets)

        _, predicted_labels = logits.max(1)
        num_hits += (targets == predicted_labels).float().sum()
        total_loss += loss * batch_size
        num_samples += batch_size

    loss = total_loss.item() / num_samples
    accuracy = num_hits.item() / num_samples
    return loss, accuracy
