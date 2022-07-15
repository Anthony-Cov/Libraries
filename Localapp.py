#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:09:15 2019

@author: antonij
"""

'''Локальная аппроксимация на несколько ходов вперед.
row - исходный временной ряд 
p - размер векторов задержек 
m - количество соседей 
fwd - горизонт прогнозирования
a0 - True/False - искать ли свободный член;
Возвращает продолжение ряда до означенного горизонта
'''
import numpy as np
import pandas as pd
import ctypes 
import Libraries
from os.path import split
from inspect import getfile
from Libraries.Util import Norm01, Metr

def LaLoV(dat, prds, predictors,p, m, fwd, a0=True):
    '''Матрица задержек'''
    '''p (задержки) можно выбирать простейшее - первый 0 автокорреляции'''
    N=len(dat)
    mada=max(dat)
    row=dat/mada
    x=[]
    for i in range(p):
        x.append(row[i:i+N-p-fwd+1])
    #x.reverse()
    X=np.matrix(x)
    '''Выбор ближайших соседей'''
    '''Взяты просто наименьшие, а можно выбирать m, чтобы убать больше ложных'''
    neib=[]
    for i in range(X.shape[1]-fwd-1):
        neib.append(np.linalg.norm(X[:,-fwd]-X[:,i]))
    n1=neib.copy()
    n1.sort()
    Xn=X[:,[neib.index(ww) for ww in n1[:m]]]
    Xn=np.flip(Xn,1)
    '''А теперь о предикторах'''
    if len(prds):
        maxlag=max([i['lag'] for i in prds])
        xx=[]
        for i,j in enumerate(prds):
            predictor=predictors[j['prd']]
            z=predictor.shift(j['lag'])[maxlag:]
            z,_,_= Norm01(z)
            xx.append(z[-Xn.shape[1]-fwd:-fwd])
        xx=np.matrix(xx)
        Xn=np.concatenate([Xn.T, xx.T], axis=1)
    else:
        Xn=Xn.T
    '''Параметры модели, наименьшие квадраты'''
    if a0:
        Xt=np.concatenate([Xn, np.ones(Xn.shape[0]).reshape(Xn.shape[0],1)], axis=1)
    else:
        Xt=Xn
    y=np.flip(X[p-1,[range(neib.index(ww)+1,neib.index(ww)+1+fwd) for ww in n1[:m]]], axis=0)
    a=np.linalg.lstsq(Xt, y, rcond=None)[0]
    '''Предсказали'''
    if len(prds):
        Xp=np.concatenate([row[-p:],  np.array(xx[:,-1]).reshape(xx.shape[0])])#.reshape(len(prds))[0]
    else:
        Xp=row[-p:]
    if a0:
        yp=np.dot(np.append(Xp,[1], axis=0),a)
    else:
        yp=np.dot(Xp,a)
    return np.array(yp).reshape(fwd)*mada
def GetDim(dat):
    nw=1000
    if len(dat)>nw: 
        y=dat[-nw:]
    else:
        y=dat
    '''C++ для размерности вложения'''
    edim = ctypes.CDLL(split(getfile(Libraries.Util))[0]+'/EmbDim.so')
    edim.EmbDim.restype = ctypes.c_int
    edim.EmbDim.argtypes = [ctypes.c_double*nw, ctypes.c_int]
    s=list(Norm01(y)[0])+[0.]*(nw-len(y))
    arr = (ctypes.c_double*nw)(*s)
    p=edim.EmbDim(arr, len(y))
    return p
def LAprExplore(dat,fwd,split,prds=[], predictors=None):
    p=GetDim(dat.data.values)
    m=(p+1)*3
    b=len(dat)-split
    x1=dat.data.values[:b]
    x_test=LaLoV(x1, prds, predictors, p, m, fwd, a0=(p>9))
    m,d1np,d2np,d3np, d4 = Metr(dat.data.values[b:b+fwd], x_test)
    return  m,d1np,d2np,d3np,d4,x_test

def LAprUse(dat, fwd, prds=[], predictors=None):
    p=GetDim(dat.data.values)
    m=(p+1)*3
    x_hat=LaLoV(dat.data.values, prds, predictors, p, m, fwd, a0=False)
    return  x_hat
