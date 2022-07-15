# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:47:20 2019
Утилиточки для прогнозирования временных рядов
@author: pop.antonij@gmail.com
"""
import numpy as np
import pandas as pd
from pymssa import MSSA
import pywt
import warnings
import matplotlib.pyplot as plt

'''Подсчет метрик: x-модель, y-реальность; выход - ошибки:средняя, абс. средняя, процентная, СКО'''
def Metr(x,y):
    y=np.array(y)
    x=np.array(x)
    d=x-y
    m=np.mean(d)
    d1=np.mean(abs(d))
    d2=sum([abs(d[i]/y[i]) for i in range(len(y)) if y[i]!=0])/len([y[i]for i in range(len(y)) if y[i]!=0 ])*100
    d3=sum([abs(d[i]/(x[i]+y[i])*2) for i in range(len(y)) if (x[i]+y[i])!=0])/len([y[i] 
                                                        for i in range(len(y)) if (x[i]+y[i])!=0 ])*100
    d4=np.std(d)
    return m, d1, d2, d3, d4

'''Расчет F1 и Accuracy для бинарного классификатора'''
def F1metr(x_pred, x_real): #классы: 1 - positive, O - negative
    x_pred, x_real= x_pred.astype(int), x_real.astype(int) 
    tp=len(np.where(x_pred[np.where(x_real==1)]==1)[0])
    tn=len(np.where(x_pred[np.where(x_real==0)]==0)[0])
    fp=len(np.where(x_pred[np.where(x_real==0)]==1)[0])
    fn=len(np.where(x_pred[np.where(x_real==1)]==0)[0])
    if (tp+fp)*(tp+fn)*tp:
        precision, recall = tp/(tp+fp), tp/(tp+fn)
        f1=2*precision*recall/(precision+recall) 
    else:
        f1=0.
    if (tp+tn+fp+fn):
        accuracy=(tp+tn)/(tp+tn+fp+fn)*100
    else:
        accuracy=0.
    return f1, accuracy

'''Линейное укладывание в диапазон [0,1], возвращает коэффициенты для восстановления (max(X))!=0'''
def Norm01(x):
    mi=np.nanmin(x)
    ma=np.nanmax(np.array(x)-mi)
    if ma>0.:
        x_n=(np.array(x)-mi)/ma
        return x_n, mi, ma
    else:
        return np.zeros(len(x)), mi, ma
'''Восстановление'''
def Nback(x_n, mi, ma):
    return x_n*ma+mi

'''скользящее среднее'''
def  MovingAverage(x, numb=10):
    n=len(x)//numb
    ma=list(x[:n])
    for j in range(len(x)-n):
        ma.append(np.mean(x[j:j+n]))
    return np.array(ma)

'''ВЧ фильтр с вейвлет-преобразованием'''
def LowPass(x, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(x)
    coeff = pywt.wavedec(x, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

'''Удаление выбросов по перцентилям'''
def remoutliers(x, tile=99.0):
    s=np.percentile(x, tile)
    y=np.delete(x, np.where(x>s))
    return y

'''Наименьшие квадраты для одной переменной'''
def MLS(x,y):
    x=np.array(x).astype(float)
    y=np.array(y).astype(float) 
    n=len(x)
    sumx, sumy=sum(x), sum(y)
    sumx2=sum([t*t for t in x])
    sumxy=sum([t*u for t,u in zip(x,y)])
    a = (n * sumxy - (sumx * sumy)) / (n * sumx2 - sumx * sumx);
    b = (sumy - a * sumx) / n;
    return a, b

'''Удаление тренда как линейной аппроксимациипервой компоненты MSSA
возвращает ряд без тренда и коэффициенты тренда Ax+b отдельно'''
def RemTrend(y):
    warnings.filterwarnings('ignore')
    mssa = MSSA(n_components=10, #'variance_threshold',
            variance_explained_threshold=0.98,#,'svht',
            window_size=None,
            verbose=False)
    x=pd.DataFrame()
    x['0']=y
    mssa.fit(x)
    tr = np.zeros(x.shape[0])
    tr[:] = np.nan
    tr = mssa.components_[0, :, :]#.sum(axis=1)
    s=tr[:,0]
    x=np.arange(len(s))
    A = np.vstack([x, np.ones(len(s))]).T
    m, c = np.linalg.lstsq(A, s, rcond=None)[0]
    z=y-m*x-c
    return z, m, c

'''Рисование раскрашенной матрицы'''
def PaintMatr(superficies, title='***', nominal=''):
    n_neighbors=superficies.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    im = ax.imshow(superficies[::-1],  interpolation='none')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=.75)
    cbar.ax.set_ylabel(nominal ,va="top", size=16 )
    cbar.set_ticks(np.linspace(superficies.min(),superficies.max(),10))
    cbar.set_ticklabels(np.linspace(superficies.min(),superficies.max(),10).astype(int))
    ax.set_xticks(np.arange(n_neighbors))
    ax.set_yticks(np.arange(n_neighbors))
    ax.set_xlabel('To node', size=16)
    ax.set_ylabel('From node', size=16)
    ax.set_facecolor('white')
    for i in range(n_neighbors):
        for j in range(n_neighbors):
            ax.text(j, i, (superficies[n_neighbors-i-1, j]).round(2),
                           ha="center", va="center", color="w", size=12)
    ax.grid()
    ax.set_xticklabels((np.arange(n_neighbors)+1).astype(str))
    ax.set_yticklabels((n_neighbors-np.arange(n_neighbors)).astype(str))
    ax.set_title(title, size=20)
    fig.tight_layout()
    plt.show()
    return 0

'''Секунды в строку ЧЧ:ММ:СС, на входе скалярно'''
def seconds_to_str(seconds):
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return "%02d:%02d:%02d" % (hh, mm, ss)
''' Строка ЧЧ:ММ:СС в число секунд. На входе вектор'''
def str_to_seconds(time):
    ss=[]
    for j in range(len(time)):
        if (time[j][0:2] == '')|(time[j][3:5] == '')|(time[j][6:8] == ''):
            ss.append(43200)
            continue
        if time[j][1:2] == ':':
            time[j]='0'+time[j]
        ss.append(int(time[j][0:2])*3600+int(time[j][3:5])*60+int(time[j][6:8]))
    return ss

''' Унитарное кодирование, выдает массив и метки'''
def one_hot(x):
    c=list(set(x))
    c.sort()
    x=np.array(x)
    a=np.zeros((len(x),len(c)))
    for n, j in enumerate(c):
        i=np.where(x==j)[0]
        a[i,n]=1
    return a.astype(int), c