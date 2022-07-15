'''Расчет характеристик временного ряда:
мера шума, размерность вложения, корреляционные размерность и энтропия,
показатель Херста, энтропия Колмогорова-синая'''
import numpy as np
import pandas as pd
import ctypes 
from os.path import split
from inspect import getfile
from scipy.stats import entropy
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

'''Размерность вложения, корреляционная и заодно оценка энтропии'''
def DimEmb(ser):
    n=len(ser)
    ent=1
    d0=0
    for k in range(2,n//2):#
        w=[]
        for i in range(n-k):
            w.append(np.array([ser[j] for j in range(i, i+k)]))
        ro=np.zeros((n-k)**2).reshape((n-k),(n-k))
        for i in range(n-k):
            for j in range(i,n-k):
                ro[i,j]=np.linalg.norm(w[i]-w[j])
        cl=[]
        cn=[]
        ls=np.linspace(ro[ro!=0].min(), ro.max(), num=20)
        for l in ls:
            c=0
            for i in range(n-k):
                for j in range(i+1,n-k):
                    c+=np.heaviside(l-ro[i,j],1)
            cn.append(c/(n-k)**2)
            cl.append(np.log(c/(n-k)**2))
        dc=(cl[1]-cl[0])/(np.log(ls[1])-np.log(ls[0]))
        if abs(dc-d0)> (ro.max() - ro.min())/50.: # dc-d0>0
            d0=dc
            ent=sum(cn)
        else:
            k-=1
            dc=d0
            ent=abs(sum(cn)/ent) #9/III-2021 abs(np.log2(sum(cn)/ent))
            break
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
    z2=[0.]*len(ser)
    z,_,_=Norm01(ser)
    z2=np.ones(len(ser)).astype(float)
    z2[np.where(z[1:]*z[:-1]!=0.)[0][1:]]=z[np.where(z[1:]*z[:-1]!=0.)][1:]/z[np.where(z[1:]*z[:-1]!=0.)][:-1]
    z2=np.log(z2)
    tau=np.arange(3,len(z))
    for t in tau:
        x=[]
        m,s=np.mean(z2[:t]),np.std(z2[:t])
        for i in range(t):
            y=[(z2[j]-m) for j in range(i)]
            x.append(sum(y))
        r=max(x)-min(x)
        h.append(np.log(r/s) if r*s > 0.  else 0.)
    h=np.array(h)
    tau=np.arange(len(z)-3)
    t=np.zeros(len(z)-3).astype(float)
    t[1:]=np.log(tau[1:]/2)
    he,b = MLS(t,h)
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
def KSEntr(data):
    l=len(data)
    e=[]
    for i in range(1,l//2+1):
        b=l//i
        hist,bins=np.histogram(data, bins=b)
        e.append(entropy(hist/l,  base=2)) #9/III-2021
    return max(e)

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