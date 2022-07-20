# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:01 2019
Тесты Грейнджера, кросс-корреляции, CCM и заодно VAR 
Данные в DataFrame dat, предикторы - predictors по колонкам
@author: pop.antonij@gmail.com
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import Libraries
from Libraries.Autoregr import VARModel
from Libraries.Util import Norm01
from Libraries.Util import Nback
from Libraries.Util import Metr
from Libraries.Util import MovingAverage

'''Тест Грейнджера'''
def GrangerTest(x1,x2, maxlag=52):
    x=pd.DataFrame()
    x['res'],_,_=Norm01(x1)
    x['cause'],_,_ =Norm01(x2)
    gr_test=sm.tsa.stattools.grangercausalitytests(x, maxlag=maxlag, verbose=False)
    ps=[gr_test[j][0]['ssr_ftest'][1] for j in gr_test]
    p1=min(ps[1:])
    l=ps.index(p1)+1
    return l, p1
'''Выбор n лучших по Грейнджеру'''    
def ChoosePredsGran(dat, predictors, n, maxlag=52):
    mindate=max(dat.week.min(),predictors.week.min())
    dat=dat.drop(dat[dat.week<mindate].index)
    predictors=predictors.drop(predictors[predictors.week<mindate].index)
    maxdate=min(dat.week.max(),predictors.week.max())
    dat=dat.drop(dat[dat.week>maxdate].index)
    predictors=predictors.drop(predictors[predictors.week>maxdate].index)
    goodpr=[]
    for i in predictors.columns[1:]:
        predictor=predictors[i]
        if predictor.dtype in (float, int):
            x=pd.DataFrame()
            x['res']=dat.data.values-MovingAverage(dat.data.values)
            x['res'],_,_=Norm01(x['res'])
            x['cause'] = predictor.values-MovingAverage(predictor.values)
            x['cause'],_,_ =Norm01(x['cause'])
            gr_test=sm.tsa.stattools.grangercausalitytests(x, maxlag=maxlag, verbose=False)
            p1=[gr_test[j][0]['ssr_ftest'][1] for j in gr_test]
            goodpr.append({'prd':i, 'score': min(p1), 'lag': p1.index(min(p1))+1})
    return sorted(goodpr, key = lambda i: i['score'])[:n]

'''Кросс-корреляция'''
def CrossCorr(datax, datay, maxlag=52):
    ccor=0
    lag=0
    dx=pd.Series(datax)
    dy=pd.Series(datay)
    for i in range(1,maxlag):
        c=abs(dx.corr(dy.shift(i),method='spearman'))
        if c>ccor:
            ccor=c
            lag=i
    return lag,ccor 
'''Выбор n лучших по кросс-корреляции''' 
def ChoosePredsCCor(dat, predictors, n, maxlag=52):
    mindate=min(dat.week.min(),predictors.week.min())
    dat=dat.drop(dat[dat.week<mindate].index)
    predictors=predictors.drop(predictors[predictors.week<mindate].index)
    goodpr=[]
    for i in predictors.columns[1:]:
        predictor=predictors[i]
        scf=max(dat.data.values)
        x1=dat.data.values/scf
        x2,_,_=Norm01(predictor)
        lag, ccor=CrossCorr(x1, x2, maxlag=maxlag)
        goodpr.append({'prd':i,'lag':lag,'score':ccor})
    return sorted(goodpr, key = lambda i: i['score'], reverse=True)[:n]

'''Хорошие по CCM'''
import skccm as ccm
from skccm.utilities import train_test_split
import warnings

def CCMTest(x1,x2, maxlag=52):
    warnings.filterwarnings("ignore")
#выбрать lag
    lag,_=CrossCorr(pd.Series(x1),pd.Series(x2) , maxlag=maxlag)
    CCM = ccm.CCM(score_metric='corrcoef') #вариант 'score'
    embed = 2
    scr=0
#выбрать embed
    for i in range(2,5):
        e1 = ccm.Embed(x1)
        e2 = ccm.Embed(x2)
        X1 = e1.embed_vectors_1d(lag,i)
        X2 = e2.embed_vectors_1d(lag,i)
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.9)
        len_tr = len(x1tr)
        lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
        sc1,sc2 = CCM.score()
        if max(sc1)>scr:
            embed, scr = i, max(sc1) 
    return(lag, embed, scr)
'''Выбор n лучших по CCM''' 
def ChoosePredsCCM(dat, predictors, n, maxlag=52):
    mindate=min(dat.week.min(),predictors.week.min())
    dat=dat.drop(dat[dat.week<mindate].index)
    predictors=predictors.drop(predictors[predictors.week<mindate].index)
    goodpr=[]
    for i in predictors.columns[1:]:
        scf=max(dat.data.values)
        x1=dat.data.values/scf
        x2,_,_=Norm01(predictors[i])
        lag, embed, score=CCMTest(x1,x2, maxlag=maxlag)
        goodpr.append({'prd':i,'lag':lag,'embed':embed, 'score':score})
    return sorted(goodpr, key = lambda i: i['score'], reverse=True)[:n]

'''Хорошие по VAR'''
def VARTest(x,y,maxlag=52):
    b=len(x)-6
    fwd=6
    score=200
    lag=0    
    for l in range(maxlag):
        x1,mi,ma=Norm01(x)
        x1=x1[l:]
        y1,_,_=Norm01(y)
        y1=pd.Series(y1).shift(l)[l:]
        vec=pd.DataFrame({'reg':x1, 'prd':y1})
        x_test=Nback(VARModel(vec[:b],fwd), mi,ma)
        d = Metr(x[b:b+fwd], x_test)
        if d[2]<score:
            score=d[2]
            lag=l
    return score, lag
'''Выбор n лучших по VAR''' 
def ChoosePredsVAR(dat, predictors, n, maxlag=52):
    mindate=min(dat.week.min(),predictors.week.min())
    dat=dat.drop(dat[dat.week<mindate].index)
    predictors=predictors.drop(predictors[predictors.week<mindate].index)
    pred=[]
    for j in predictors.columns[1:]:
        prd=predictors[j]
        s,l=VARTest(dat.data.values, prd,maxlag=maxlag)
        pred.append({'prd':j,'lag':l,'score':s})
    pred=sorted(pred, key = lambda i: i['score'], reverse=False)
    return pred[:n]