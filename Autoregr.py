# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:40:00 2019
Прогнозирование временного ряда VAR
векторная авторегрессия
@author: pop.antonij@gmail.com
"""
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from Libraries.Util import Metr, Norm01, Nback

def VARModel(x,fwd):
    data = list()
# contrived dataset with dependency    
    for i in range(len(x)):
        row = list(x.iloc[i, :])
        data.append(row)
# fit model
    model = VAR(data)
    model_fit = model.fit(maxlags=12, method='ols', ic=None, trend='ct',)
# make prediction
    yhat = model_fit.forecast(model_fit.y, steps=fwd)
    del model
    return yhat[:,0]

def VARExplore(dat, fwd,split, prds=[], predictors=None):
    b=len(dat)-split
    d, mi,ma=Norm01(dat.data.values)
    x=GetDataVec(d,prds, predictors)
    x1=x[:b]
    x_test=Nback(VARModel(x1,fwd), mi,ma)
    m,d1np,d2np,d3np,d4 = Metr(x_test, dat.data.values[b:b+fwd])
    return  m,d1np,d2np,d3np,d4, x_test

def VARUse(dat, fwd, prds=[], predictors=None):
    d, mi,ma=Norm01(dat)
    x=GetDataVec(d, prds, predictors)
    x_pred=Nback(VARModel(x,fwd), mi,ma)
    return x_pred

def GetDataVec(d, prds, predictors):
    if len(prds):
        x=pd.DataFrame()
        maxlag=max([i['lag'] for i in prds])
        x['reg']=d[maxlag:]
        for i,j in enumerate(prds):
            predictor=predictors[j['prd']]
            xx=predictor.shift(j['lag'])[maxlag:]
            x['prd'+str(i)]=Norm01(xx.values)[0]
    else:
        x=pd.DataFrame({'reg':d,'2':d})
    return x
