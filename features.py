'''Расчет характеристик временного ряда:
мера шума, размерность вложения, корреляционные размерность и энтропия,
показатель Херста, энтропия Колмогорова-синая'''
import numpy as np
import pandas as pd
import ctypes 
from os.path import split
from inspect import getfile
from scipy.stats import entropy
from scipy.linalg import hankel
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import warnings
import Libraries
import gc
from Libraries.Util import Norm01, Nback, MLS 


'''мера шумности по ско разностей к ско ряда. (больше мера - меньше шума)'''
def NoiseFactor(data, axis=0, ddof=1):
    a = Norm01(data)[0]
    m = np.std(pd.Series(a).diff().dropna().abs())
    sd = a.std(axis=axis, ddof=ddof)
    return 1-float(np.where(sd == 0, 0, m/sd))

'''Оценка случайного блуждания'''
def RandWalk(ser):
    return abs(NoiseFactor(pd.Series(ser).diff().fillna(method='bfill')))

'''Размерность вложения по корреляционному интегралу'''
def DimEmb(tser, eps=.1):
    ser,_,_=Norm01(tser)
    n=len(ser)
    cn=[1]
    d0=0
    h=hankel(ser)
    for k in range(2,n//2):#
        ent=sum(cn)
        w=h[:n-k, :k]
        ro=np.zeros([n-k, n-k])
        for i,j in combinations(np.arange(n-k), 2):
            norm=np.linalg.norm(w[i]-w[j])
            ro[i,j]=norm
            ro[j,i]=norm
        cl=[]
        cn=[]
        ls=np.linspace(ro[ro!=0].min(), ro.max(), num=20)
        for l in ls:
            c=np.heaviside(l-ro-np.diag(np.ones(n-k)),1).sum()//2
            cn.append(c/(n-k)**2)
            cl.append(np.log(c/(n-k)**2))
        dc=(cl[1]-cl[0])/(np.log(ls[1])-np.log(ls[0]))
        if abs(dc-d0) > eps: # (ro.max() - ro.min())/50.: 
            d0=dc
        else:
            break
    k-=1
    dc=d0
    ent=sum(cn)/ent
    return k, dc, ent #k - размерность вложения, dc - корреляционная размерность, ent - оценка энтропии.

def CEmbDim(dat): #То же по-быстрому с C++ процедурой EmbDim.so'
    nw=1000
    if len(dat)>1000:
        y=dat[-1000:]
    else:
        y=dat
    emd = ctypes.CDLL(split(getfile(Libraries.Util))[0]+'/EmbDim.so')
    emd.EmbDim.restype = ctypes.c_int
    emd.EmbDim.argtypes = [ctypes.c_double*nw, ctypes.c_int]
    s=list(Norm01(y)[0])+[0.]*(nw-len(y))
    arr = (ctypes.c_double*nw)(*s)
    return emd.EmbDim(arr, len(y))

def CCorrent(dat): #Корреляционная энтропия по-быстрому с C++ процедурой CorrEntr.cpp
    nw=1000
    if len(dat)>1000:
        y=dat[-1000:]
    else:
        y=dat
    emd = ctypes.CDLL(split(getfile(Libraries.Util))[0]+'/CorrEntr.so')
    emd.CorrEntr.restype = ctypes.c_double
    emd.CorrEntr.argtypes = [ctypes.c_double*nw, ctypes.c_int]
    s=list(Norm01(y)[0])+[0.]*(nw-len(y))
    arr = (ctypes.c_double*nw)(*s)
    centr=emd.CorrEntr(arr, len(y))
    del(arr)
    del(emd)
    ctypes._reset_cache()
    gc.collect()
    return centr


'''Показатель Хёрста (R/S и H траектории)'''
def HurstTraj(ser): #RS-trajectory of Hurst
    h=[]
    z2,_,_=Norm01(ser)
    tau=np.arange(3,len(z2))
    for t in tau:
        m,s=np.mean(z2[:t]), np.std(z2[:t])
        x=(z2[:t]-m).cumsum()
        r=max(x)-min(x)
        h.append(np.log(r/s) if r*s > 0.  else 0.)
    h=np.array(h)
    t=np.array([0.])
    t=np.concatenate([t, np.log(tau[1:]/2)])
    l=int(len(t)/50)
    he,b = MLS(t[:l],h[:l])
    mem=np.where([(h[i+1]-h[i])<0. for i in range(len(h)-1)])[0]
    mem=mem[0] if len(mem) else 0
    return t,h,he,mem #t-ln(tau); h - R/S trajectory (Hurst's tr=h/t); he - Hurst's exponent; mem - series' memory

def CHurst(dat): #То же по-быстрому с C++ процедурой HurstExp.so
    nw=1000
    if len(dat)>1000:
        y=dat[-1000:]
    else:
        y=dat
    he = ctypes.CDLL(split(getfile(Libraries.Util))[0]+'/HurstExp.so')
    he.HurstExp.restype = ctypes.c_double
    he.HurstExp.argtypes = [ctypes.c_double*nw, ctypes.c_int]
    s=list(y)+[0.]*(nw-len(y))
    arr = (ctypes.c_double*nw)(*s)
    return he.HurstExp(arr, len(y))

'''Kolmogorov-Sinai Entropy'''
def ksent(time_series, k=5, tau=1, eps=1e-10):
    """
    Вычисляет энтропию Колмогорова-Синая для временного ряда.

    Параметры:
    - time_series: одномерный временной ряд (numpy array).
    - k: количество ближайших соседей (по умолчанию 3).
    - tau: временной лаг для построения вложений (по умолчанию 1).
    - eps: маленькое значение для избежания деления на ноль (по умолчанию 1e-10).

    Возвращает:
    - ks_entropy: оценка энтропии Колмогорова-Синая.
    """
    n = len(time_series)
    m = n - (k - 1) * tau  # количество вложений
    # Создаем вложения (embedding)
    embeddings = np.array([time_series[i:i + k * tau:tau] for i in range(m)])
    # Используем KDTree для поиска ближайших соседей
    tree = KDTree(embeddings)
    distances = []
    for i in range(m):
        # Ищем k+1 ближайших соседей (включая саму точку)
        dist, _ = tree.query(embeddings[i], k + 1)
        distances.append(dist[-1])  # берем расстояние до k-го соседа
    distances = np.array(distances)
    # Вычисляем KS энтропию
    ks_entropy = np.mean(np.log(distances + eps)) / tau
    return ks_entropy

'''Колмогоровская сложность по оценке Лемпеля — Зива'''
def LempelZiv(S):
    n=len(S)
    i=0
    C=u=v=vmax=1
    while (u+v)<n:
        if S[i+v] == S[u+v]:
            v+=1
        else:
            vmax = max(v, vmax)
            i+=1
            v=1
            if i==u:
                C+=1
                u+=vmax
                i=0
                vmax=v
            else:
                v=1
    if v!=1:
        C+=1
    return C

'''энтропия ряда по Шеннону'''
def ShEntr(data, bin=25):
    hist,bins=np.histogram(data, bins=bin)
    return entropy(hist/len(data),  base=2)

'''Всё вместе в словарь'''
def get_features(ser):
    warnings.filterwarnings('ignore')
    features={}
    features['noise']=NoiseFactor(ser, axis=0, ddof=1)
    features['hurst']=СНurst(ser) #HurstTraj(ser)[2]
    features['coent']=DimEmb(ser)[2]
    features['ksent']=KSEntr(ser)
    features['randm']=RandWalk(ser)
    return features
