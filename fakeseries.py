from random import choice
import numpy as np
def fake(l, func='', noise=.5, season=.5):
    x=np.linspace(-10,10,l)
    if func=='log':#logistic
        y=1/(1+np.exp(x))
    elif func=='exp': #exponent
        x+=7.
        y=np.ones(l)
        y[np.where(x>=0)]=np.exp(-x[np.where(x>=0)]/3)
    elif func=='gau':
        y=1/np.exp(x**2/10)
    elif func=='rwa':
        y=[1.]
        z=np.linspace(-1,1,10)
        for i in range(l-1):
            y+=[y[i]+choice(z)]
        y=np.array(y)
    else:
        y=np.ones(l)
    y+=np.random.random(l)*noise-noise/2
    y+=np.sin(x*5)*season
    return(y)