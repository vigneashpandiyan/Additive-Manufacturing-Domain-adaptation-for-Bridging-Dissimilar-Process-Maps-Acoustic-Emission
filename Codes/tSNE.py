"""
Created on Fri Jan  5 10:50:03 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the topics on domain adaptation
--> Monitoring Of Laser Powder Bed FusionProcess By Bridging Dissimilar Process MapsUsingDeep Learning-based Domain Adaptation onAcoustic Emissions

@any reuse of this code should be authorized by the first owner, code author
"""

# %% Libraries to import

from Visualization_Utils import *
from matplotlib.pyplot import specgram
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
np.random.seed(1974)

# %%


def plot_embeddings(folder, tsne_fit, group, graph_name1, graph_title, limits, xlim=None, ylim=None):

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]

    group = np.ravel(group)
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')

    uniq = list(set(df['label']))
    uniq = np.sort(uniq)

    print(uniq)

    marker = ["o", "*", ">", "o", "*", ">"]
    color = ['#dd221c', '#16e30b', 'blue', '#fab6b6', '#a6ffaa', '#b6befa']

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(12, 9), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)

    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)

    plt.ylim(b1, b2)
    plt.xlim(a1, a2)

    # for i in uniq:
    for i in range(len(uniq)):

        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        plt.plot(a, b, color=color[i], label=uniq[i], marker=marker[i], linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=20, fontsize=25)
    plt.ylabel('Dimension 2', labelpad=20, fontsize=25)
    plt.title(graph_title, fontsize=30)

    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False)
    plt.savefig(os.path.join(folder, graph_name1), bbox_inches='tight', dpi=100)
    plt.show()


def TSNEplot(folder, output, target, graph_name1, graph_name2, graph_title, limits, perplexity):

    # array of latent space, features fed rowise

    output = np.array(output)
    target = np.array(target)

    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ', perplexity)

    group = target
    group = np.ravel(group)

    RS = np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]
    x3 = tsne_fit[:, 2]

    df = pd.DataFrame(dict(x=x1, y=x2, z=x3, label=group))

    groups = df.groupby('label')
    uniq = list(set(df['label']))
    print(uniq)
    uniq = np.sort(uniq)
    # uniq=["0","1","2","3"]

    plot_embeddings(folder, tsne_fit, target, graph_name1,
                    graph_title, limits, xlim=None, ylim=None)

    marker = ["o", "*", ">", "o", "*", ">"]
    color = ['#dd221c', '#16e30b', 'blue', '#fab6b6', '#a6ffaa', '#b6befa']

    fig = plt.figure(figsize=(20, 8), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=20)
    ax = plt.axes(projection='3d')

    ax.grid(False)
    ax.view_init(elev=15, azim=110)  # 115

    ax.set_facecolor('white')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)
    c1, c2 = detect_limits(x3, limits)

    ax.set_ylim(b1, b2)
    ax.set_zlim(c1, c2)
    ax.set_xlim(a1, a2)
    for i in range(len(uniq)):

        print(i)
        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        c = x3[indx]

        ax.plot(a, b, c, color=color[i], label=uniq[i], marker=marker[i], linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=20, fontsize=15)
    plt.ylabel('Dimension 2', labelpad=20, fontsize=15)
    ax.set_zlabel('Dimension 3', labelpad=20, fontsize=15)
    plt.title(graph_title, fontsize=20)

    plt.locator_params(nbins=6)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(os.path.join(folder, graph_name2), bbox_inches='tight', dpi=200)
    plt.show()

    return ax, fig


def plot_embeddings_1(folder, tsne_fit, group, graph_name1, graph_title, limits, xlim=None, ylim=None):

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]

    group = np.ravel(group)
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')

    uniq = list(set(df['label']))
    uniq = np.sort(uniq)

    print(uniq)

    marker = ["o", "*", ">", "o", "*", ">"]
    color = ['#dd221c', '#16e30b', 'blue', '#fab6b6', '#a6ffaa', '#b6befa']

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(12, 9), dpi=100)

    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)

    plt.ylim(b1, b2)
    plt.xlim(a1, a2)

    # for i in uniq:
    for i in range(len(uniq)):

        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        plt.plot(a, b, color=color[i], label=uniq[i], marker=marker[i], linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=20, fontsize=25)
    plt.ylabel('Dimension 2', labelpad=20, fontsize=25)
    plt.title(graph_title, fontsize=30)

    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False)
    plt.savefig(os.path.join(folder, graph_name1), bbox_inches='tight', dpi=100)
    plt.show()


def TSNEplot_1(folder, output, target, graph_name1, graph_name2, graph_title, limits, perplexity):

    # array of latent space, features fed rowise

    output = np.array(output)
    target = np.array(target)

    print('output shape: ', output.shape)
    print('target shape: ', target.shape)
    print('perplexity: ', perplexity)

    group = target
    group = np.ravel(group)

    RS = np.random.seed(123)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]
    x3 = tsne_fit[:, 2]

    marker = ["o", "*", ">", "o", "*", ">"]
    color = ['#dd221c', '#16e30b', 'blue', '#fab6b6', '#a6ffaa', '#b6befa']

    df = pd.DataFrame(dict(x=x1, y=x2, z=x3, label=group))

    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq = np.sort(uniq)
    plot_embeddings_1(folder, tsne_fit, target, graph_name1,
                      graph_title, limits, xlim=None, ylim=None)

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(20, 8), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=20)
    ax = plt.axes(projection='3d')

    ax.grid(False)
    ax.view_init(elev=15, azim=110)  # 115

    ax.set_facecolor('white')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)
    c1, c2 = detect_limits(x3, limits)

    ax.set_ylim(b1, b2)
    ax.set_zlim(c1, c2)
    ax.set_xlim(a1, a2)
    for i in range(len(uniq)):

        print(i)
        indx = (df['label']) == uniq[i]

        a = x1[indx]
        b = x2[indx]
        c = x3[indx]

        ax.plot(a, b, c, color=color[i], label=uniq[i], marker=marker[i], linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=20, fontsize=15)
    plt.ylabel('Dimension 2', labelpad=20, fontsize=15)
    ax.set_zlabel('Dimension 3', labelpad=20, fontsize=15)
    plt.title(graph_title, fontsize=20)

    plt.locator_params(nbins=6)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(os.path.join(folder, graph_name2), bbox_inches='tight', dpi=200)
    plt.show()

    return ax, fig


# %%


def detect_limits(scores_normal, limits):
    # find q1 and q3 values
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    Threshold0 = normal_avg - (normal_std * limits)
    Threshold1 = normal_avg + (normal_std * limits)
    return Threshold0, Threshold1
