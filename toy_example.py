# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:08:10 2021

@author: Filipe Fortuna
"""

import numpy as  np
from matplotlib import pyplot as plt
import pandas as pd
from pyesn import ESN
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from statsmodels.graphics.tsaplots import plot_acf

# configurações do Python
plt.close('all')

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

# esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs)

esn = ESN(n_inputs=n_inputs, n_outputs=n_outputs, Rsize=202,
          Sparsity=0.274, Spectral_radius=0.95,
          alpha=0.2786, Ridge=0.0001, Noise_level=0.001)

#%% Usando a rede  

yhat = esn.train(inputs_train, outputs_train)

ypredict = esn.predict(inputs_test)

NRMSE_treino = np.sum(np.sqrt(np.mean((yhat - outputs[0:ind,:])**2)/np.var(outputs[0:ind,:])))
NRMSE_teste  =  np.sum(np.sqrt(np.mean((ypredict - outputs[ind:,:])**2)/np.var(outputs[ind:,:])))

desempenho = lambda ytarget,ypred: (r2_score(ytarget,ypred), mean_squared_error(ytarget,ypred), explained_variance_score(ytarget,ypred))

print('\nTraining: R2: %1.2f  MSE: %1.2e  ExpVar: %1.2f' %(desempenho(outputs_train,yhat)),'RMSE: %1.2e' %(NRMSE_treino))

print('\nTesting: R2: %1.2f  MSE: %1.2e  ExpVar: %1.2f' %(desempenho(outputs_test,ypredict)),'RMSE: %1.2e' %(NRMSE_teste))


#%% saídas gráficas


plt.plot(t,outputs,'.k', alpha = 0.80,label="target system")
plt.plot(t[:ind], yhat,'b', label="trained ESN")
plt.plot(t[ind:], ypredict,'--r', label="free running ESN")
plt.legend()
plt.tight_layout()
plt.show()

# residuo de estimação
res = outputs_train - yhat
t_train = t[:ind]

fig, [ax1,ax2]= plt.subplots(2, figsize=(12,6))

ax1.plot(t_train,res, '.')
ax1.set_xlabel(r'$ time, t $')
ax1.set_ylabel(r'$ residue,y_1-\hat y_1 $')

# correlograma do resíduo (somente da etapa de treinamento)
plot_acf(res,ax=ax2)
plt.tight_layout()
plt.show()


