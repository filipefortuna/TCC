# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:48:01 2021

@author: Filipe Fortuna
"""

import warnings 
import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
from pyesn import ESN
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

#%% configurações do Python

warnings.filterwarnings("ignore", category=FutureWarning)
plt.close('all')

#%% Estraindo dados

df = pd.read_csv("cstr.dat", sep="\s+", index_col=None)
df.columns = ['time','qc','Ca','T']

t = df['time']
inputs = df[['qc']].to_numpy()
outputs = df[['Ca','T']].to_numpy()


#%% Imlementando a ESN
n_inputs =  1
n_outputs = 2

Train_fraction = 0.7

indice = int(len(inputs)*Train_fraction)

inputs_train = inputs[:indice,:]
outputs_train = outputs[:indice,:]
inputs_test = inputs[indice:,:]
outputs_test = outputs[indice:,:]


#%% Treinando a rede

esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs, Rsize=202,
          Sparsity=0.274, Spectral_radius=0.95,
          alpha=0.2786, Ridge=0.0001, Noise_level=0.001)

#%% Usando a rede  

yhat = esn.train(inputs_train, outputs_train)

ypredict = esn.predict(inputs_test)

NRMSE_treino = np.sum(np.sqrt(np.mean((yhat - outputs[0:indice,:])**2)/np.var(outputs[0:indice,:])))
NRMSE_teste  =  np.sum(np.sqrt(np.mean((ypredict - outputs[indice:,:])**2)/np.var(outputs[indice:,:])))

desempenho = lambda y,ypred: (r2_score(y,ypred), mean_squared_error(y,ypred))

print('\nTraining: R2: %1.2f  MSE: %1.2e' %(desempenho(outputs_train,yhat)),'RMSE: %1.2e' %(NRMSE_treino))

print('\nTesting: R2: %1.2f  MSE: %1.2e' %(desempenho(outputs_test,ypredict)),'RMSE: %1.2e' %(NRMSE_teste))

#%% saídas gráficas
fig, [ax1,ax2] = plt.subplots(1,2, figsize=(12,6), sharex=True)

ax1.plot(t,outputs[:,0],'.k', alpha = 0.80,label="target system")
ax1.plot(t[:indice], yhat[:,0],'b', label="trained ESN")
ax1.plot(t[indice:], ypredict[:,0],'--r', label="free running ESN")
ax1.set_ylabel(r'$ Output, y_1 $')
ax1.set_xlabel(r'$ time, t $')
ax1.legend()

ax2.plot(t,outputs[:,1],'.k', alpha = 0.80,label="target system")
ax2.plot(t[:indice], yhat[:,1],'b', label="trained ESN")
ax2.plot(t[indice:], ypredict[:,1],'--r', label="free running ESN")
ax2.set_ylabel(r'$ Output, y_2 $')
ax2.set_xlabel(r'$ time, t $')
ax2.legend()

plt.tight_layout()
plt.show()

res = outputs_train - yhat

fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, figsize=(12,6))

ax1.plot(t[:indice],res[:,0], '.')
ax1.set_xlabel(r'$ time, t $')
ax1.set_ylabel(r'$ residue,y_1-\hat y_1 $')

ax2.plot(t[:indice],res[:,1],'.')
ax2.set_ylabel(r'$ residue,y_2-\hat y_2 $')
ax2.set_xlabel(r'$ time, t $')

# correlograma do resíduo (somente da etapa de treinamento)
plot_acf(res[:,0],ax=ax3)
plot_acf(res[:,1],ax=ax4)
plt.tight_layout()
plt.show()