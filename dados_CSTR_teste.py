# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:11:56 2021

@author: Filipe Fortuna
"""

import time
import warnings 
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# configurações do Python
warnings.filterwarnings("ignore", category=FutureWarning)
plt.close('all')
np.set_printoptions(precision=4, suppress=True, floatmode='maxprec')

#%%

""" 
Determinar o modelo linearizado do processo relacionando 
as mudanças nas variáveis de estado (Ca e T), com as
variáveis de entrada (q e qc).

"""

#%% Geração de dados

# importar modelo do processo
from model_CSTR import CSTR_modificado

# importar gerador de sinal
from PySinal import SinalSequencia

# Condição nominal
xs=  [8.36e-2, 440.2]
us=  [100.0, 103.42]

q_mean = us[0] # médida de q
qc_mean = us[1] # média de qc

num_samples = 1000 # número de amostras 
samples = range(num_samples)

# geração do sinal galsiano pseudo randomico
lim_porc = 0.07
delta_q = lim_porc*q_mean
delta_qc = lim_porc*qc_mean

q_range = [q_mean - delta_q, q_mean + delta_q]
qc_range = [qc_mean - delta_qc, qc_mean + delta_qc]

Ts = 0.15
b_range = [10*Ts,300*Ts] # Obs: x*Ts>0.5

q = SinalSequencia(num_samples, q_range, b_range, type='RGS', random_state=14111999)
qc = SinalSequencia(num_samples, qc_range, b_range, type='RGS', random_state=30031991)


# condição nominal
x0 = xs

# Intervalo de Integração
tspan = [0,Ts]
t = np.arange(0,num_samples*Ts,Ts)

# criando vetores para armazenar os dados de saida gerados
Ca = np.zeros(num_samples)
T = np.zeros(num_samples)

inicio = time.time()

for i in samples:
    u0 = [q[i], qc[i]]
    modelo_nlinear = lambda t,x: CSTR_modificado(t, x, u0)
    
    tspan = [i*Ts,(i+1)*Ts]

    sol = solve_ivp(modelo_nlinear, tspan, x0, method='BDF')
    x = sol.y
    Ca[i] = x[0][-1]
    T[i] = x[1][-1]
    
    x0 = [Ca[i], T[i]]

fim = time.time()
tempo = fim-inicio
print("tempo do reator", tempo)
#%% Plotagem dos dados

lim = num_samples
if num_samples>99:
    lim = 100

fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, figsize=(13,6.5))

ax1.step(t[:lim],q[:lim], '-')
ax1.set_ylabel(r'$ entrada, q $')

ax2.step(t[:lim],qc[:lim],'-')
ax2.set_ylabel(r'$ entrada, qc $')

ax3.step(t[:lim],Ca[:lim], '-')
ax3.set_xlabel(r'$ tempo $')
ax3.set_ylabel(r'$ saida, Ca $')

ax4.step(samples[:lim],T[:lim],'-')
ax4.set_xlabel(r'$ tempo $')
ax4.set_ylabel(r'$ saida, T $')

#%% Armazenando dados em um Excel

dados = {'t':list(t), 'q':list(q), 'qc':list(qc), 'Ca':list(Ca), 'T':list(T)}
dados = pd.DataFrame(data=dados)
dados.to_excel('dados_CSTR_teste.xls', index=False)
