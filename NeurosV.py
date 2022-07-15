# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:40:00 2019
Прогнозирование временного ряда с векторным выходом LSTM
@author: pop.antonij@gmail.com
"""
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Model
from keras.layers import concatenate, Dropout
from keras.layers import Input, LSTM, Dense
from keras.backend import clear_session

from Libraries.Util import Norm01, Metr
from Libraries.features import CEmbDim

'''ПРОГНОЗИРОВАНИЕ'''
'''Здесь с векторным выходом.!!!'''
'''Модель LSTM теперь сделаем так (это мультипредикторная версия!!!
X2, Y2 - списки массивов предикторов)'''
def LSTMModelV(X1, Y1, X2, Y2, fwd): #Обучение
    epo=300
    predam=len(X2)
    nopred=(len(X2)==0)
    clear_session()
    main_input = Input(shape=(X1.shape[1:]), name='main_input')
    lstm_out = LSTM(128, return_sequences=True)(main_input)
    if nopred:
        x=lstm_out
        minp=[main_input]
    else:
        x=lstm_out
        minp=[main_input]
        for i in range(predam): #доп. входы для каждого предиктора
            aux_input = Input(shape=(X2[i].shape[1:]), name='aux_input'+str(i))
            x = concatenate([x, aux_input])
            minp+=[aux_input]
    lstm_out0 = LSTM(128, return_sequences=True)(x)
    do1=Dropout(.5)(lstm_out0)
    lstm_out1 = LSTM(128, return_sequences=False)(do1)
    main_output = Dense(Y1.shape[1], activation='linear', name='main_output')(lstm_out1)
    model = Model(inputs=minp, outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'], loss_weights=[1.])
    if nopred:
        model.fit([X1], Y1, validation_data=([X1], Y1), epochs=epo, batch_size=6, verbose=0)
        prediction_test = model.predict([X1])
    else:       
        X=[X1[fwd:]]+[np.array(X2[j]) for j in range(predam)]    #собираем предикторы для обучения 
        model.fit(X, Y1[fwd:], epochs=epo, batch_size=6, verbose=0)
        prediction_test = model.predict(X)
    return prediction_test[:,-1].reshape(prediction_test.shape[0]), model
'''Это мультипредикторная версия с векторным выходом!!!'''
def LSTMForcastV(model, dat, p, predictors, fwd, length): #Предсказание
    predam=len(p)
    nopred=(len(p)==0)
    scf=max(dat)
    x=dat[-length:].values/scf
    X1=x.reshape(1,1,length)
    if nopred:
        X2=np.array([[]]).reshape(0,1)
        Y2=np.array([[]]).reshape(0,1)
        y = model.predict([X1])[0]
    else: 
        X2=[]
        Y2=[]
        for j in p:    
            X, Y=GetPredictorV(predictors[j['prd']], fwd, length)
            X2.append(X[fwd:])
            Y2.append(Y[fwd:])
        X=[X1[-1:]]+[np.array(X2[j][-1:]) for j in range(predam)]    #собираем предикторы для обучения 
        y = model.predict(X)
    return (y*scf).reshape(y.shape[-1])
'''Исследование LSTM split - метсто разделения на обучающую-тестовую,
от конца ряда.'''
def LSTMExploreV(dat, fwd, split, p=[], predictors=None): #Тестирование
    d=dat.data
    predam=len(p)
    b=len(d)-split
    X1,Y1, scf, length=GetDataV(d[:b], fwd)
    if not predam:
        X2=np.array([[]]).reshape(0,1)
        Y2=np.array([[]]).reshape(0,1)
        prediction_test, model=LSTMModelV(X1, Y1, X2, Y2, fwd)
    else:
        X2=[]
        Y2=[]
        for j in p:    
            X, Y=GetPredictorV(predictors[j['prd']], fwd, length) ###<----### 7/IX-2021
            X2.append(X[fwd:-split])
            Y2.append(Y[fwd:-split])
        prediction_test, model=LSTMModelV(X1, Y1, X2, Y2,fwd)
    y_pred=LSTMForcastV(model, d[:b], p, predictors, fwd, length)
    m,d1np,d2np,d3np,d4 = Metr(y_pred, d[b:b+fwd])
    y=np.concatenate((prediction_test*scf, d[b-1:b], y_pred), axis=None)
    return m,d1np,d2np,d3np,d4,y
'''Предсказание через мультипредикторную LSTM'''
def LSTMUseV(dat, fwd, prds=[], predictors=None): #Истользование
    d=dat.data
    predam=len(p)
    X1,Y1, scf, length=GetDataV(d,fwd)
    if not predam:
        X2=np.array([[]]).reshape(0,1)
        Y2=np.array([[]]).reshape(0,1)
        prediction_test, model=LSTMModelV(X1, Y1, X2, Y2, fwd)
    else:
        X2=[]
        Y2=[]
        for j in p:    
            X, Y=GetPredictorV(predictors[j['prd']], fwd, length)
            X2.append(X[fwd:])
            Y2.append(Y[fwd:])
        prediction_test, model=LSTMModelV(X1, Y1, X2, Y2,fwd)
    y=LSTMForcastV(model, d, p, predictors, fwd, length)
    return y
'''Подготовка данных data - Series'''
def GetDataV(data, fwd):
    length=2*CEmbDim(data)+1
    l=len(data)-length-fwd #(-2) для старых предикторов
    scf=max(data)
    X=data[:-fwd]/scf
    y=data.values[:]/scf
    Y=[y[i:i+fwd] for i in range(len(data)-fwd)]
    data_gen = TimeseriesGenerator(X, Y, length=length, sampling_rate=1, batch_size=1)
    X1 = [data_gen[i][0] for i in range(l)]
    Y1 = [data_gen[i][1] for i in range(l)]
    X1=np.array(X1)
    Y1=np.array(Y1)
    return X1, Y1.reshape(Y1.shape[0],Y1.shape[2]), scf, length
'''Подготовка предикторов из таблицы'''
def GetPredictorV(prd, fwd, length):
    predictor=prd.shift(fwd)[fwd:]
    l=len(predictor)-length
    Xp,_,_ = Norm01(predictor)
    Yp,_,_ = Norm01(predictor)
    data_gen = TimeseriesGenerator(Xp, Yp, length=length, sampling_rate=1, batch_size=1)
    X2 = [data_gen[i][0] for i in range(l)]
    Y2 = [data_gen[i][1] for i in range(l)]
    X2=np.array(X2)
    Y2=np.array(Y2)
    return X2, Y2
