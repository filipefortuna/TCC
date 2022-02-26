# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:33:02 2021

@author: Filipe Fortuna
"""

import warnings 
import numpy as  np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# configurações do Python
warnings.filterwarnings("ignore", category=FutureWarning)
plt.close('all')
np.set_printoptions(precision=4, suppress=True, floatmode='maxprec')

#%%

""" 
Encontrar o tempo que o reator chega no estado estacionário
quando é feito mudanças nas variáveis de estado (Ca e T), com as
variáveis de entrada (q e qc).

"""

#%% Geração de dados

# importar modelo do processo
from model_CSTR import CSTR_modificado


# Condição nominal
x0=  [8.36e-2, 440.2]
u0=  [100.0, 103.42]

# testes de variação nas entradas
u0[0] = u0[0] - 0.07*u0[0] # q
u0[1] = u0[1] + 0.07*u0[1] # qc

# intervalo de integração
t_final = 10
tspan = [0, t_final] # [t_inicial, t_final]


# Instanciar o modelo
modelo_nlinear = lambda t,x: CSTR_modificado(t, x, u0)

# solver: integrador
sol = solve_ivp(modelo_nlinear, tspan, x0, method='BDF', t_eval = np.linspace(*tspan,100))

# resultado
t = sol.t # tempo
x = sol.y; # Ca, T
y = x.T;

yfinal1 = y[-1,0]
yfinal2 = y[-1,1]
erro1 = 100*abs((yfinal1-y[:,0])/yfinal1)
erro2 = 100*abs((yfinal2-y[:,1])/yfinal2)
N = len(t)
ind = 0
for i in range(N):
    if erro1[i]<0.01 and erro2[i]<0.01:
        ind = i
        break
        
print(f"\nErro menor que 1% em t = {t[ind] :1.2f}")
print(f'Erro = {erro1[ind]: 1.2f}')
print(f'Erro = {erro2[ind]: 1.2f}')

#%% Plotagem dos dados

# Instanciar uma figura
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(12,6))

ax1.plot(t,x[0,:], 'r-', lw=2)
ax1.set_ylabel('Ca, mol/L', fontsize=14)
ax1.set_xlabel('t, min', fontsize=14)
ax1.set_title('Concentração vs Tempo',fontsize=14)

ax2.plot(t,x[1,:],'b-', lw=2)
ax2.set_ylabel('T, K', fontsize=14)
ax2.set_xlabel('t, min', fontsize=14)
ax2.set_title('Temperatura vs Tempo',fontsize=14)

# Ajustar o preenchimento entre os subplots
plt.tight_layout()
plt.show()

# salvar uma figura
plt.savefig('Step_CSTR.tif')


