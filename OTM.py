# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 09:55:54 2022

@author: Filipe Fortuna
"""
import warnings 
import numpy as  np
import pandas as pd
from pyesn import ESN
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution

# configurações do Python
warnings.filterwarnings("ignore", category=FutureWarning)

#%% Lendo os dados

data = pd.read_excel('dados_toyexample.xls', index_col=None)

t = data['t']
inputs = data[['uk']].to_numpy()
outputs = data[['yk']].to_numpy()


#%% Configurando a rede
n_inputs =  1
n_outputs = 1

Train_fraction = 0.7

ind = int(len(inputs)*Train_fraction)

inputs_train = inputs[:ind,:]
outputs_train = outputs[:ind,:]
inputs_test = inputs[ind:,:]
outputs_test = outputs[ind:,:]


#%% Treinando a rede

def minimo(x):

    x0 = int(x[0])
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]

    esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs, Rsize=x0,
          Sparsity=x1, Spectral_radius=x2,
          alpha=x3, Ridge=x4, Noise_level=0.001)

    # Treinando a rede
    yhat = esn.train(inputs_train, outputs_train)
    ypredict = esn.predict(inputs_test)
    
    desempenho = lambda y,ypred: (r2_score(y,ypred))
    a = desempenho(outputs_test,ypredict)
    print("Obtive R^2 = ",a)
    U = (a-1)**2
    print("Função minimo = ",U)
    return U

bounds=[(190,210),(0.1,0.4),(0.8,0.95),(0.1,0.4),(0.0001,0.001)]
result = differential_evolution(minimo, bounds, tol=0.01 )
print('xopt_DE =',result.x)
print('fopt_DE =',result.fun)