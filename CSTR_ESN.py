# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:44:00 2021

@author: Filipe Fortuna
"""

import time
import warnings 
import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
from pyesn import ESN
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

# configurações do Python
warnings.filterwarnings("ignore", category=FutureWarning)
plt.close('all')

#%% Lendo os dados
data = pd.read_excel('dados_CSTR.xls', index_col=None)

t = data['t']
inputs = data[['q','qc']].to_numpy()
outputs = data[['Ca','T']].to_numpy()


#%% Configurando a rede
n_inputs =  2
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

fig = plt.subplots(1,1, figsize=(13.5,6.5), sharex=True)
plt.plot(t,outputs[:,0],'.k', alpha = 0.80,label="target system")
plt.plot(t[:indice], yhat[:,0],'b', label="trained ESN")
plt.plot(t[indice:], ypredict[:,0],'--r', label="free running ESN")
plt.ylabel(r'$ Output, y_1 $')
plt.xlabel(r'$ time, t $')
plt.legend()

fig = plt.subplots(1,1, figsize=(13.5,6.5), sharex=True)
plt.plot(t,outputs[:,1],'.k', alpha = 0.80,label="target system")
plt.plot(t[:indice], yhat[:,1],'b', label="trained ESN")
plt.plot(t[indice:], ypredict[:,1],'--r', label="free running ESN")
plt.ylabel(r'$ Output, y_2 $')
plt.xlabel(r'$ time, t $')
plt.legend()

plt.tight_layout()
plt.show()

# residuo de estimação
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

# Figura reduzida com apenas 100 amostras 
lim = 100
fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, figsize=(13,6.5))

ax1.step(t[:lim],inputs[:lim,0], '-')
ax1.set_ylabel(r'$ entrada, q $')

ax2.step(t[:lim],inputs[:lim,1],'-')
ax2.set_ylabel(r'$ entrada, qc $')

ax3.step(t[:lim],outputs[:lim,0], '-b')
ax3.step(t[:lim],yhat[:lim,0], '.k')
ax3.set_ylabel(r'$ saida, Ca $')

ax4.step(t[:lim],outputs[:lim,1], '-b')
ax4.step(t[:lim],yhat[:lim,1], '.k')
ax4.set_ylabel(r'$ saida, T $')


#%% teste com novos dados

print('\n')
print('-------------------------------------------------')
print('Teste da rede com dados distintos do treinamnto')


data = pd.read_excel('dados_CSTR_teste.xls', index_col=None)

t = data['t']
inputs = data[['q','qc']].to_numpy()
outputs = data[['Ca','T']].to_numpy()


# predição dos dados de saida pela ESN com contagem de tempo
inicio = time.time()
ypredict = esn.predict(inputs, continuation=False)
fim = time.time()
tempo = fim-inicio
print("\nTempo da ESN = ", tempo)

# avaliação dos resultados obtidos
NRMSE_teste  =  np.sum(np.sqrt(np.mean((ypredict - outputs)**2)/np.var(outputs)))

desempenho = lambda y,ypred: (r2_score(y,ypred), mean_squared_error(y,ypred))
print('\nTesting: R2: %1.2f  MSE: %1.2e' %(desempenho(outputs,ypredict)),'RMSE: %1.2e' %(NRMSE_teste))


# saidas graficas
fig, [ax1,ax2] = plt.subplots(1,2, figsize=(12,6), sharex=True)

ax1.plot(t,outputs[:,0],'.k', alpha = 0.80,label="target system")
ax1.plot(t, ypredict[:,0],'--r', label="free running ESN")
ax1.set_ylabel(r'$ Output, y_1 $')
ax1.set_xlabel(r'$ time, t $')
ax1.legend()

ax2.plot(t,outputs[:,1],'.k', alpha = 0.80,label="target system")
ax2.plot(t, ypredict[:,1],'--r', label="free running ESN")
ax2.set_ylabel(r'$ Output, y_2 $')
ax2.set_xlabel(r'$ time, t $')
ax2.legend()

plt.tight_layout()
plt.show()

