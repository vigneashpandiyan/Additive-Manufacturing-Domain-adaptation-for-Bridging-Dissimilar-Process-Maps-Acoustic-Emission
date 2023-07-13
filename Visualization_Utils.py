# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:21:51 2022

@author: srpv
"""

#%% Libraries to import

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
plt.rcParams.update(plt.rcParamsDefault)

#%%
def detect_limits(scores_normal,limits):
    # find q1 and q3 values
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    Threshold0 = normal_avg - (normal_std * limits)
    Threshold1 = normal_avg + (normal_std * limits)
    return Threshold0,Threshold1

def plot_embeddings(tsne_fit, group,graph_name1,graph_title,limits, xlim=None, ylim=None):
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    
    
    group = np.ravel(group)
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')
    
    
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    
    print(uniq)
    
    
    marker= ["o","*",">","o","*",">"]
    color = [ '#dd221c', '#16e30b', 'blue','#fab6b6','#a6ffaa','#b6befa']
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(12,9), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    
    a1,a2=detect_limits(x1,limits)
    b1,b2=detect_limits(x2,limits)
    
    plt.ylim(b1, b2)
    plt.xlim(a1, a2)
    
    
    # for i in uniq:
    for i in range(len(uniq)):
        
        indx = (df['label']) == uniq[i]
        a=x1[indx]
        b=x2[indx]
        plt.plot(a, b, color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=10)
        
        
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=25)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=25)
    plt.title(graph_title,fontsize = 30)
    
    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(bbox_to_anchor=(1, 1),frameon=False)
    plt.savefig(graph_name1, bbox_inches='tight',dpi=100)
    plt.show()


def TSNEplot(output,target,graph_name1,graph_name2,graph_title,limits,perplexity):
    
    #array of latent space, features fed rowise
    
    output = np.array(output)
    target = np.array(target)
    
    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ',perplexity)
    
    group=target
    group = np.ravel(group)
    
    RS=np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)
    
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    x3=tsne_fit[:, 2]
    
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=group))
   
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    print(uniq)
    uniq=np.sort(uniq)
    # uniq=["0","1","2","3"]

    
    plot_embeddings(tsne_fit, target,graph_name1,graph_title,limits, xlim=None, ylim=None)
    
    marker= ["o","*",">","o","*",">"]
    color = [ '#dd221c', '#16e30b', 'blue','#fab6b6','#a6ffaa','#b6befa']
    
    fig = plt.figure(figsize=(20,8), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=20)
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(elev=15,azim=110)#115
    
    
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    a1,a2=detect_limits(x1,limits)
    b1,b2=detect_limits(x2,limits)
    c1,c2=detect_limits(x3,limits)
    
    ax.set_ylim(b1,b2)
    ax.set_zlim(c1,c2)
    ax.set_xlim( a1,a2)
    for i in range(len(uniq)):
        
        print(i)
        indx = (df['label']) == uniq[i]
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        
        ax.plot(a, b, c ,color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=10)
       
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=15)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=15)
    ax.set_zlabel('Dimension 3',labelpad=20,fontsize=15)
    plt.title(graph_title,fontsize = 20)
    
    
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name2, bbox_inches='tight',dpi=200)
    plt.show()
    
    return ax,fig
#%%



def plot_embeddings_1(tsne_fit, group,graph_name1,graph_title,limits, xlim=None, ylim=None):
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    
    
    group = np.ravel(group)
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')
    
    
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    
    print(uniq)
    
    marker= ["o","*",">","o","*",">"]
    color = [ '#dd221c', '#16e30b', 'blue','#fab6b6','#a6ffaa','#b6befa']
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(12,9), dpi=100)
    
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    a1,a2=detect_limits(x1,limits)
    b1,b2=detect_limits(x2,limits)
    
    plt.ylim(b1, b2)
    plt.xlim(a1, a2)
   
    
    # for i in uniq:
    for i in range(len(uniq)):
        
        indx = (df['label']) == uniq[i]
        a=x1[indx]
        b=x2[indx]
        plt.plot(a, b, color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=10)
        
        
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=25)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=25)
    plt.title(graph_title,fontsize = 30)
    
    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(bbox_to_anchor=(1, 1),frameon=False)
    plt.savefig(graph_name1, bbox_inches='tight',dpi=100)
    plt.show()



def TSNEplot_1(output,target,graph_name1,graph_name2,graph_title,limits,perplexity):
    
    #array of latent space, features fed rowise
    
    output = np.array(output)
    target = np.array(target)
    
    
    print('output shape: ', output.shape)
    print('target shape: ', target.shape)
    print('perplexity: ',perplexity)
    
    group=target
    group = np.ravel(group)
    
    RS=np.random.seed(123)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)
    
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    x3=tsne_fit[:, 2]
    
    marker= ["o","*",">","o","*",">"]
    color = [ '#dd221c', '#16e30b', 'blue','#fab6b6','#a6ffaa','#b6befa']
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=group))
   
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    plot_embeddings_1(tsne_fit, target,graph_name1,graph_title,limits, xlim=None, ylim=None)
    
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(20,8), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=20)
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(elev=15,azim=110)#115
    
    
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    
    
    a1,a2=detect_limits(x1,limits)
    b1,b2=detect_limits(x2,limits)
    c1,c2=detect_limits(x3,limits)
    
    ax.set_ylim(b1,b2)
    ax.set_zlim(c1,c2)
    ax.set_xlim( a1,a2)
    for i in range(len(uniq)):
        
        print(i)
        indx = (df['label']) == uniq[i]
        
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        
        ax.plot(a, b, c ,color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=10)
        
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=15)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=15)
    ax.set_zlabel('Dimension 3',labelpad=20,fontsize=15)
    plt.title(graph_title,fontsize = 20)
    
    
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name2, bbox_inches='tight',dpi=200)
    plt.show()
    
    return ax,fig


#%%

def plot_log(logs,Training_loss_mean,Training_associative_loss_mean):
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7,5))
    plt.plot(logs['Epoch'], logs['D1_logloss'], label='D1 validation loss', marker='o', c='red',linewidth =1.5)
    plt.plot(logs['Epoch'], logs['D2_logloss'], label='D2 validation loss', marker='*', c='blue',linewidth =1.5)
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.title('Validation loss',fontsize=20);
    plt.legend( loc='best',fontsize=15,frameon=False)
    plt.savefig('Validation loss.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6,4))
    plt.plot(logs['Epoch'], logs['D1_accuracy'], label='D1 Testset', marker='o', c='red',linewidth = 1.5)
    plt.plot(logs['Epoch'], logs['D2_accuracy'], label='D2 Testset', marker='*',  c='blue',linewidth = 1.5)
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.title('Accuracy',fontsize=20);
    plt.legend( loc='best',fontsize=15,frameon=False)
    plt.savefig('Accuracy.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7,5))
    plt.plot(logs['Epoch'], logs['Classification_loss'],'blue', marker='o', label='Training loss',linewidth =1.5)
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.title('DA Training loss',fontsize=20);
    plt.legend( loc='upper right',fontsize=15,frameon=False)
    plt.savefig('Training loss.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7,5))
    plt.plot(logs['Epoch'], logs['Associative_loss'], label='Associative loss',marker='o', c='green',linewidth =1.5)
    plt.plot(logs['Epoch'], logs['Regularizer_loss'], label='Regularizer loss',marker='*', c='purple',linewidth =1.5)
    plt.xlabel('Iteration',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.title('Domain adaptation losses',fontsize=20);
    plt.legend( loc='upper right',fontsize=15,frameon=False)
    plt.savefig('Domain adaptation losses.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7,5))
    plt.plot(logs['Epoch'], logs['learning_rate'], label='learning rate', c='black',linewidth =1.5)
    plt.xlabel('Iteration',fontsize=20)
    plt.ylabel('Learning rate',fontsize=20)
    plt.title('Domain adaptation losses',fontsize=20);
    plt.legend( loc='upper right',fontsize=15,frameon=False)
    plt.savefig('lr Domain adaptation losses.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6,4))
    fig, ax = plt.subplots()
    plt.rc('font', size=15)
    plt.plot(Training_loss_mean, label='Cross-entropy loss', marker='^', c='black',linewidth =1.5,markerfacecolor='red', markersize=7,linestyle='dashed')
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Loss',fontsize=15)
    plt.title('Cross-entropy loss',fontsize=15);
    plt.legend( loc='best',fontsize=15,frameon=False)
    plt.savefig('Domain Adaptation Training_loss_mean.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6,4))
    fig, ax = plt.subplots()
    plt.rc('font', size=15)
    plt.plot(Training_associative_loss_mean, label='Associative loss', marker='*', c='black',linewidth =1.5,markerfacecolor='green', markersize=7)
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Loss',fontsize=15)
    plt.title('Associative loss',fontsize=15);
    plt.legend( loc='best',fontsize=15,frameon=False)
    plt.savefig('Domain Adaptation Associative loss_mean.png', dpi=600,bbox_inches='tight')
    plt.show()

#%%

def predict(dataset,model,model_name):
    X, y = [], []
    
    
    model.load_state_dict(torch.load(model_name)) #'./just_D1.pth'
    model.eval()
    model = model[0]  # only embedding

    for image, label in dataset:
        x = model(image.cuda())
        X.append(x.detach().cpu().numpy())
        y.append(label.detach().cpu().numpy())
        
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]
    
    return X, y


#%%

def ThreeDplot(filename,S1,L1,check):
    
    if check=='D1':
        df2 = pd.DataFrame(L1) 
        df2.columns = ['Categorical']
        df2=df2['Categorical'].replace(2,3)
        df2 = pd.DataFrame(df2)
        df2=df2['Categorical'].replace(1,2)
        df2 = pd.DataFrame(df2)
        df2=df2['Categorical'].replace(0,1)
        df2 = pd.DataFrame(df2)
        L1 = df2.to_numpy() 
        L1=np.ravel(L1) 
        print(L1)
    else:
        df2 = pd.DataFrame(L1) 
        df2.columns = ['Categorical']
        df2=df2['Categorical'].replace(0,4)
        df2 = pd.DataFrame(df2)
        df2=df2['Categorical'].replace(1,5)
        df2 = pd.DataFrame(df2)
        df2=df2['Categorical'].replace(2,6)
        df2 = pd.DataFrame(df2)
        L1 = df2.to_numpy()
        L1=np.ravel(L1) 
        print(L1)
    
    S1, _, L1, _ = train_test_split(S1, L1, test_size=0.90, random_state=66)
    
    graph_name1= str(filename)+'_2D'+'.png'
    graph_name2= str(filename)+'_3D'+'.png'
    
    ax,fig=TSNEplot(S1,L1,graph_name1,graph_name2,str(filename),limits=2.5,perplexity=10)
    
    graph_name= str(filename)+'.gif'
   
    return ax,fig,graph_name


def ThreeComparisonPlot(filename, S1, L1, S2, L2):
    
    df2 = pd.DataFrame(L1) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(2,3)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,2)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(0,1)
    df2 = pd.DataFrame(df2)
    L1 = df2.to_numpy()
    
    

    df2 = pd.DataFrame(L2) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,4)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,5)
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,6)
    df2 = pd.DataFrame(df2)
    L2 = df2.to_numpy()
    
    S1=np.array(S1)
    L1=np.ravel(L1) 
    
    
    S2=np.array(S2)
    L2=np.ravel(L2)
    
    S3=np.concatenate((S1,S2), axis=0)
    L3=np.concatenate((L1,L2), axis=0)
    
    S3, _, L3, _ = train_test_split(S3, L3, test_size=0.90, random_state=66)
    
    graph_name1= str(filename)+'_2D'+'.png'
    graph_name2= str(filename)+'_3D'+'.png'
    
    ax,fig=TSNEplot_1(S3,L3,graph_name1,graph_name2,str(filename),limits=2.5,perplexity=30)
    
    graph_name= str(filename)+'.gif'
   
    
    return ax,fig,graph_name

