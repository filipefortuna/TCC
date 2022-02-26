# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:13:30 2021

@author: Filipe Fortuna
"""
import numpy as  np
from matplotlib import pyplot as plt
import pandas as pd

# configurações do Python
plt.close('all')


#%% Gerando sinais de entrada e saida

# numero de amostras
samples = 1000

# criando vetores para armazenar os dados
y = np.zeros(samples)
u = np.zeros(samples)

for k in range(samples-1):
    if k <= 500:
        uk = np.sin(np.pi*k/125)
    if k > 500:
        uk = 0.8*np.sin(np.pi*k/125) +0.2*np.sin(2*np.pi*k/25)
    u[k] = uk
    
    y[k+1] =  (y[k]*y[k-1]*y[k-2]*(y[k-2]-1)*u[k-1]+u[k])/(1+y[k-1]**2+y[k-2]**2)
    
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,6), sharex=True)

t = np.arange(samples)

ax1.plot(t,u, '-r',label="u_k")
ax1.set_ylabel(r'$ Entrada, u_k $')
ax1.set_xlabel(r'$ time, t $')
ax1.legend()
ax2.plot(t,y, '-b',label="y_k")
ax2.set_ylabel(r'$ Saída, y_k $')
ax2.set_xlabel(r'$ time, t $')
ax2.legend()
plt.show()

#%% Armazenando dados em um Excel

dados = {'t':list(t), 'uk':list(u), 'yk':list(y)}
dados = pd.DataFrame(data=dados)
dados.to_excel('dados_toyexample.xls', index=False)
