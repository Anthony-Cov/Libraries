#from __future__ import division
import numpy as np
import pandas as pd
import networkx as nx
import Libraries
from Libraries.Util import Norm01
import ctypes 
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from random import choice
import matplotlib
import matplotlib.pyplot as plt
from os.path import split
from inspect import getfile

class Graph():
    def __init__(self, row):
        nw=1000
        '''C++ для размерности вложения'''
        edim = ctypes.CDLL(split(getfile(Libraries.Util))[0]+'/EmbDim.so')
        edim.EmbDim.restype = ctypes.c_int
        edim.EmbDim.argtypes = [ctypes.c_double*nw, ctypes.c_int]
        s=list(Norm01(row)[0])+[0.]*(nw-len(row))
        arr = (ctypes.c_double*nw)(*s)
        self.demb=edim.EmbDim(arr, len(row))
        self.row=row

    '''Создание графа из ряда'''
    def MakeGraph(self, mindist=.01, method='hierarchy', k=25):
        depth=self.demb*2+1
        self.traject=pd.DataFrame(columns=['step']+['t-'+str(j) for j in range(depth-1,-1,-1)])
        self.traject['step']=np.arange(len(self.row)-depth)
        for i in range(len(self.row)-depth):
            self.traject.loc[i, self.traject.columns[1:]]=self.row[i:i+depth]
        if method=='hierarchy':
            '''матрица связей для кластеризации'''
            link = linkage(self.traject[self.traject.columns[1:]], 'ward', 'euclidean')
            '''оптимальное расстояние для кластеризации'''
            dist = link[:, 2]
            dist_rev = dist[::-1]
            i=np.where(np.abs(dist_rev[:-5]-dist_rev[5:])<(max(dist_rev)*mindist))[0][0]
            d=dist_rev[i]
            '''кластеризация'''
            self.traject['cluster']=fcluster(link, d, criterion='distance')
        else:
            model = KMeans(n_clusters=k)
            zz=self.traject[self.traject.columns[1:]]
            model.fit(zz)
            '''кластеризация'''
            self.traject['cluster']=model.labels_
        self.traject['cluster']=[list(self.traject['cluster'].drop_duplicates().values).index(i)+1
                                 for i in self.traject['cluster']]
        '''Граф'''
        nod=self.traject['cluster'].values
        self.G=nx.DiGraph()
        self.G.add_nodes_from(self.traject['cluster'].unique())
        r=[(nod[j], nod[j+1]) for j in range(len(self.traject)-1)]
        c=[r.count(i) for i in r]
        self.G.add_weighted_edges_from([(nod[j], nod[j+1],c[j]) for j in range(len(self.traject)-1)])
        self.n_nodes=self.G.number_of_nodes()
        return self.G
    '''Расчет энтропии графа'''
    def GraphEntropy(self):
        centrality = nx.degree_centrality(self.G).values()
        centrality = np.asarray(list(centrality))
        centrality /= centrality.sum()
        dist = np.asarray(centrality)
        ent = np.nansum( dist *  np.log2( 1/dist ) )
        return ent
    '''Восстановление ряда из графа'''
    def Restore(self, l):
        nd=1
        restored=[]
        adj=dict(self.G.adjacency())
        while len(restored)<l:
            restored.append(nd)
            variety=[]
            for k in adj[nd].keys():
                variety+=[k]*adj[nd][k]['weight']
            nd=choice(variety)
        recall=[]
        for j in restored:
            recall.append(choice(self.traject[self.traject.cluster==j]['t-0'].values))
        return recall
    '''Рисование графа'''
    def DrawGraph(self):
        matplotlib.style.use('fast')
        fig=plt.figure(figsize=(10, 10))
        lnd=self.n_nodes+1
        nnd=lnd//7
        nlist=[range(i,i+lnd//nnd) for i in range(0,lnd-lnd//nnd,lnd//nnd)]+[range(lnd-lnd//nnd,lnd)]
        nx.draw(self.G, pos=nx.shell_layout(self.G, nlist=nlist),
                       with_labels=True,
                       node_color='pink', font_weight='bold',font_size=10,
                       edge_color='b',node_size=300, width=2 )
        #nx.spectral_layout(G), random, circular
        fig.tight_layout()
        plt.show()
        return self
    '''Матрица смежности'''
    def AdjMatr(self):
        adj=dict(self.G.adjacency())
        n=self.n_nodes
        matr=np.zeros((n,n))
        for i in adj.keys():
            for j in adj[i].keys():
                matr[i-1,j-1]=adj[i][j]['weight']
        return matr
    '''Матрица вероятностей переходов'''
    def TransMatr(self, paint=False):
        matplotlib.style.use('fast')
        '''Вероятности переходов по всем узлам в матрицу'''
        n_neighbors=self.n_nodes
        adj=dict(self.G.adjacency())
        superficies=np.zeros([n_neighbors,n_neighbors])
        for i in adj.keys():
            if len(adj[i].keys())==0:
                   continue
            for j in adj.keys():
                if i==j:
                    if j in adj[i].keys():
                        s=sum([adj[i][k]['weight'] for k in adj[i].keys()])
                        superficies[i-1,j-1]=adj[i][j]['weight']/s if s>0. else 0.
                else:
                    probab=0
                    for path in nx.algorithms.simple_paths.all_simple_paths(self.G, i, j):
                        p=1
                        for n in range(len(path)-1):
                            s=sum([adj[path[n]][k]['weight'] for k in adj[path[n]].keys()])
                            if s>0.:
                                p*=adj[path[n]][path[n+1]]['weight']/s
                                if p<.001: break # 01.04.2021
                            else:
                                p=0.
                                break
                        probab+=p
                    superficies[i-1,j-1]=probab
            superficies[i-1,:]=superficies[i-1,:]/superficies[i-1,:].sum()
            '''Нарисовать heatmap'''
        if paint:
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.patch.set_facecolor('white')
            im = ax.imshow(superficies[::-1],  interpolation='none')
            cbar = ax.figure.colorbar(im, ax=ax, shrink=.75)
            cbar.ax.set_ylabel('Probability (%)' ,va="top", size=16 )
            cbar.set_ticks(np.linspace(superficies.min(),superficies.max(),10))
            cbar.set_ticklabels(np.linspace(superficies.min()*100,superficies.max()*100,10).astype(int))
            ax.set_xticks(np.arange(n_neighbors))
            ax.set_yticks(np.arange(n_neighbors))
            ax.set_xlabel('To node', size=16)
            ax.set_ylabel('From node', size=16)
            ax.set_facecolor('white')
            for i in range(n_neighbors):
                for j in range(n_neighbors):
                    ax.text(j, i, (100*superficies[n_neighbors-i-1, j]).round(2),
                                   ha="center", va="center", color="w", size=12)
            ax.grid()
            ax.set_xticklabels((np.arange(n_neighbors)+1).astype(str))
            ax.set_yticklabels((n_neighbors-np.arange(n_neighbors)).astype(str))
            ax.set_title("Transition probability between graph nodes", size=20)
            fig.tight_layout()
            plt.show()
        return superficies

'''Создание графа из ряда'''
def MakeGraph(row, mindist=.01, method='hierarchy', k=25):
    nw=1000
    '''C++ для размерности вложения'''
    edim = ctypes.CDLL(split(getfile(Libraries.Util))[0]+'/EmbDim.so')
    edim.EmbDim.restype = ctypes.c_int
    edim.EmbDim.argtypes = [ctypes.c_double*nw, ctypes.c_int]
    s=list(Norm01(row)[0])+[0.]*(nw-len(row))
    arr = (ctypes.c_double*nw)(*s)
    demb=edim.EmbDim(arr, len(row))
    depth=demb*2+1
    traject=pd.DataFrame(columns=['step']+['t-'+str(j) for j in range(depth-1,-1,-1)])
    traject['step']=np.arange(len(row)-depth)
    for i in range(len(row)-depth):
        traject.loc[i, traject.columns[1:]]=row[i:i+depth]
    if method=='hierarchy':
        '''матрица связей для кластеризации'''
        link = linkage(traject[traject.columns[1:]], 'ward', 'euclidean')
        '''оптимальное расстояние для кластеризации'''
        dist = link[:, 2]
        dist_rev = dist[::-1]
        i=np.where(np.abs(dist_rev[:-5]-dist_rev[5:])<(max(dist_rev)*mindist))[0][0]
        d=dist_rev[i]
        '''кластеризация'''
        traject['cluster']=fcluster(link, d, criterion='distance')
    else:
        model = KMeans(n_clusters=k)
        zz=traject[traject.columns[1:]]
        model.fit(zz)
        '''кластеризация'''
        traject['cluster']=model.labels_
    traject['cluster']=[list(traject['cluster'].drop_duplicates().values).index(i)+1 for i in traject['cluster']]
    '''Граф'''
    nod=traject['cluster'].values
    G=nx.DiGraph()
    G.add_nodes_from(traject['cluster'].unique())
    r=[(nod[j], nod[j+1]) for j in range(len(traject)-1)]
    c=[r.count(i) for i in r]
    G.add_weighted_edges_from([(nod[j], nod[j+1],c[j]) for j in range(len(traject)-1)])
    return G
'''Расчет энтропии графа'''
def GraphEntropy(G):
    centrality = nx.degree_centrality(G).values()
    centrality = np.asarray(list(centrality))
    centrality /= centrality.sum()
    dist = np.asarray(centrality)
    ent = np.nansum( dist *  np.log2( 1/dist ) )
    return ent
'''Матрица смежности'''
def AdjMatr(G, paint=True):
    adj=dict(G.adjacency())
    n=G.number_of_nodes()
    matr=np.zeros((n,n))
    for i in adj.keys():
        for j in adj[i].keys():
            matr[i-1,j-1]=adj[i][j]['weight']
    if paint:
        n=G.number_of_nodes()
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('white')
        im = ax.imshow(matr[::-1],  interpolation='none')
        cbar = ax.figure.colorbar(im, ax=ax, shrink=.75)
        cbar.ax.set_ylabel('Weight' ,va="top", size=16 )
        cbar.set_ticks(np.linspace(matr.min(),matr.max(),10))
        cbar.set_ticklabels(np.linspace(matr.min(),matr.max(),10).astype(int))
        ax.set_xticks(np.arange(n)+.5)
        ax.set_yticks(np.arange(n)+.5)
        ax.set_xlabel('To node', size=16)
        ax.set_ylabel('From node', size=16)
        ax.set_facecolor('white')
        for i in range(n):
            for j in range(n):
                ax.text(j, i, (matr[n-i-1, j]).astype(int),
                               ha="center", va="center", color="w", size=12)
        ax.grid()
        ax.set_xticklabels((np.arange(n)+1).astype(str), {'horizontalalignment': 'right'})
        ax.set_yticklabels((n-np.arange(n)).astype(str), {'verticalalignment': 'bottom'})
        ax.set_title("Graph adjacensy", size=20)
        fig.tight_layout()
        plt.show()
    return matr
'''Матрица переходов'''
def TransMatr(G, paint=True):
    matplotlib.style.use('fast')
    '''Вероятности переходов по всем узлам в матрицу'''
    n_neighbors=len(G.nodes())
    adj=dict(G.adjacency())
    superficies=np.zeros([n_neighbors,n_neighbors])
    for i in adj.keys():
        if len(adj[i].keys())==0:
               continue
        for j in adj.keys():
            if i==j:
                if j in adj[i].keys():
                    s=sum([adj[i][k]['weight'] for k in adj[i].keys()])
                    superficies[i-1,j-1]=adj[i][j]['weight']/s if s>0. else 0.
            else:
                probab=0
                for path in nx.algorithms.simple_paths.all_simple_paths(G, i, j):
                    p=1
                    for n in range(len(path)-1):
                        s=sum([adj[path[n]][k]['weight'] for k in adj[path[n]].keys()])
                        if s>0.:
                            p*=adj[path[n]][path[n+1]]['weight']/s
                            if p<.001: break # 01.04.2021
                        else:
                            p=0.
                            break
                    probab+=p
                superficies[i-1,j-1]=probab
        superficies[i-1,:]=superficies[i-1,:]/superficies[i-1,:].sum() #Нормируем
    '''Нарисовать heatmap'''
    if paint:
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('white')
        im = ax.imshow(superficies[::-1],  interpolation='none')
        cbar = ax.figure.colorbar(im, ax=ax, shrink=.75)
        cbar.ax.set_ylabel('Probability (%)' ,va="top", size=16 )
        cbar.set_ticks(np.linspace(superficies.min(),superficies.max(),10))
        cbar.set_ticklabels(np.linspace(superficies.min()*100,superficies.max()*100,10).astype(int))
        ax.set_xticks(np.arange(n_neighbors)+.5)
        ax.set_yticks(np.arange(n_neighbors)+.5)
        ax.set_xlabel('To node', size=16)
        ax.set_ylabel('From node', size=16)
        ax.set_facecolor('white')
        for i in range(n_neighbors):
            for j in range(n_neighbors):
                ax.text(j, i, (100*superficies[n_neighbors-i-1, j]).round(2),
                               ha="center", va="center", color="w", size=12)
        ax.grid()
        ax.set_xticklabels((np.arange(n_neighbors)+1).astype(str), {'horizontalalignment': 'right'})
        ax.set_yticklabels((n_neighbors-np.arange(n_neighbors)).astype(str), {'verticalalignment': 'bottom'})
        ax.set_title("Transition probability between graph nodes", size=20)
        fig.tight_layout()
        plt.show()
    return superficies
'''Рисование графа shell-layout'''
def DrawGraph(G, title=''):
    fig=plt.figure(figsize=(10, 10))
    plt.title(title, size=20)
    lnd=G.number_of_nodes()+1
    nnd=lnd//7
    nlist=[range(i,i+lnd//nnd) for i in range(0,lnd-lnd//nnd,lnd//nnd)]+[range(lnd-lnd//nnd,lnd)]
    nx.draw(G, pos=nx.shell_layout(G, nlist=nlist),
                   with_labels=True,
                   node_color='pink', font_weight='bold',font_size=10,
                   edge_color='b',node_size=300, width=2 )
    #nx.spectral_layout(G), random, circular
    fig.tight_layout()
    return fig
